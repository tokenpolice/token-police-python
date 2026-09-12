"""Image_size resolution cascade (request → response → binary → default).

Mapper turns parseable WxH into image_output_pixels. Extractors must:
  - prefer request dims when present
  - read response metadata / in-memory binary headers when request omitted
  - apply conservative provider defaults only as last resort
  - never throw (GOLDEN RULE); never invent for xAI without binary
"""
from __future__ import annotations

import base64
import struct
import unittest
import zlib

from token_police.enforcer import (
    _extract_google_imagen,
    _extract_hf_image,
    _extract_openai_images,
    _extract_together_image,
    _extract_xai_image,
    _image_dims_from_binary,
    _image_size_from_dims,
    _parse_image_size_str,
    _resolve_image_size,
)


def _make_png(w: int, h: int) -> bytes:
    """Minimal valid RGB PNG with given dimensions."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + (b"\xff\x00\x00" * w) for _ in range(h))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


PNG_2X3 = _make_png(2, 3)
PNG_2X3_B64 = base64.b64encode(PNG_2X3).decode("ascii")
PNG_512X256 = _make_png(512, 256)
PNG_512X256_B64 = base64.b64encode(PNG_512X256).decode("ascii")


class TestHelpers(unittest.TestCase):
    def test_image_size_from_dims(self):
        self.assertEqual(_image_size_from_dims(512, 768), "512x768")
        self.assertEqual(_image_size_from_dims(None, 768), "")
        self.assertEqual(_image_size_from_dims(512, 0), "")

    def test_parse_image_size_str(self):
        self.assertEqual(_parse_image_size_str("1024x1024"), "1024x1024")
        self.assertEqual(_parse_image_size_str("512×768"), "512x768")
        self.assertEqual(_parse_image_size_str("auto"), "")
        self.assertEqual(_parse_image_size_str("16:9"), "")
        self.assertEqual(_parse_image_size_str(""), "")
        self.assertEqual(_parse_image_size_str(None), "")

    def test_binary_png(self):
        self.assertEqual(_image_dims_from_binary(PNG_2X3), "2x3")
        self.assertEqual(_image_dims_from_binary(PNG_512X256), "512x256")
        self.assertEqual(_image_dims_from_binary(b"not-an-image"), "")
        self.assertEqual(_image_dims_from_binary(b""), "")
        self.assertEqual(_image_dims_from_binary(None), "")

    def test_resolve_priority_request_wins(self):
        result = type("R", (), {"data": [{"b64_json": PNG_2X3_B64}]})()
        self.assertEqual(
            _resolve_image_size(
                request_size="1024x1792",
                result=result,
                provider="openai",
                model="dall-e-3",
            ),
            "1024x1792",
        )

    def test_resolve_binary_before_default(self):
        result = type("R", (), {"data": [{"b64_json": PNG_512X256_B64}]})()
        self.assertEqual(
            _resolve_image_size(
                request_size="auto",
                result=result,
                provider="openai",
                model="gpt-image-1",
            ),
            "512x256",
        )

    def test_resolve_together_default(self):
        self.assertEqual(
            _resolve_image_size(provider="together", model="flux"),
            "1024x1024",
        )

    def test_resolve_xai_no_default(self):
        self.assertEqual(
            _resolve_image_size(provider="xai", model="grok-imagine-image", allow_default=False),
            "",
        )


class TestTogetherImageSize(unittest.TestCase):
    def test_captures_width_height(self):
        out = _extract_together_image(
            (),
            {"model": "black-forest-labs/FLUX.1-schnell", "n": 1, "width": 512, "height": 768},
            type("R", (), {"data": [{}], "usage": None})(),
        )
        self.assertEqual(out["items"]["images_generated"], 1)
        self.assertEqual(out["items"]["image_size"], "512x768")

    def test_default_when_omitted(self):
        out = _extract_together_image(
            (),
            {"model": "m", "n": 1},
            type("R", (), {"data": [{}], "usage": None})(),
        )
        self.assertEqual(out["items"]["image_size"], "1024x1024")

    def test_binary_overrides_default(self):
        out = _extract_together_image(
            (),
            {"model": "m", "n": 1},
            type("R", (), {"data": [{"b64_json": PNG_2X3_B64}], "usage": None})(),
        )
        self.assertEqual(out["items"]["image_size"], "2x3")

    def test_width_only_incomplete_falls_to_default(self):
        out = _extract_together_image(
            (),
            {"model": "m", "width": 1024},
            type("R", (), {"data": [{}], "usage": None})(),
        )
        # incomplete request dims → default (not partial invent)
        self.assertEqual(out["items"]["image_size"], "1024x1024")


class TestHfImageSize(unittest.TestCase):
    def test_parameters_width_height(self):
        out = _extract_hf_image(
            (),
            {
                "model": "black-forest-labs/FLUX.1-dev",
                "parameters": {"width": 640, "height": 480},
            },
            None,
        )
        self.assertEqual(out["items"]["image_size"], "640x480")

    def test_pil_size_from_result(self):
        pil = type("Img", (), {"size": (320, 240)})()
        out = _extract_hf_image((), {"model": "m"}, pil)
        self.assertEqual(out["items"]["image_size"], "320x240")

    def test_default_when_no_dims(self):
        out = _extract_hf_image((), {"model": "m"}, None)
        self.assertEqual(out["items"]["image_size"], "1024x1024")


class TestGoogleImagenSize(unittest.TestCase):
    def test_config_width_height(self):
        cfg = type("C", (), {"width": 1024, "height": 1024, "number_of_images": 1})()
        result = type("R", (), {"generated_images": [{}], "images": None})()
        out = _extract_google_imagen((), {"model": "imagen-3.0-generate-002", "config": cfg}, result)
        self.assertEqual(out["items"]["image_size"], "1024x1024")

    def test_default_when_only_count(self):
        cfg = type("C", (), {"number_of_images": 2})()
        result = type("R", (), {"generated_images": [{}, {}], "images": None})()
        out = _extract_google_imagen((), {"model": "imagen-3.0-generate-002", "config": cfg}, result)
        self.assertEqual(out["items"]["images_generated"], 2)
        self.assertEqual(out["items"]["image_size"], "1024x1024")

    def test_aspect_only_no_invent(self):
        cfg = type("C", (), {"number_of_images": 1, "aspect_ratio": "16:9"})()
        result = type("R", (), {"generated_images": [{}], "images": None})()
        out = _extract_google_imagen((), {"model": "imagen-3.0-generate-002", "config": cfg}, result)
        self.assertEqual(out["items"]["image_size"], "")


class TestOpenAIImageSize(unittest.TestCase):
    def test_request_size(self):
        out = _extract_openai_images(
            (),
            {"model": "dall-e-3", "size": "1792x1024", "n": 1},
            type("R", (), {"data": [{}], "usage": None})(),
        )
        self.assertEqual(out["items"]["image_size"], "1792x1024")

    def test_b64_when_auto(self):
        out = _extract_openai_images(
            (),
            {"model": "gpt-image-1", "size": "auto", "n": 1},
            type("R", (), {"data": [{"b64_json": PNG_512X256_B64}], "usage": None})(),
        )
        self.assertEqual(out["items"]["image_size"], "512x256")

    def test_default_when_omitted(self):
        out = _extract_openai_images(
            (),
            {"model": "dall-e-3", "n": 1},
            type("R", (), {"data": [{"url": "https://example.com/x.png"}], "usage": None})(),
        )
        self.assertEqual(out["items"]["image_size"], "1024x1024")


class TestXaiImageSize(unittest.TestCase):
    def test_no_default_url_only(self):
        # Simulates ImageResponse where .base64 raises (format=url)
        class _Img:
            @property
            def base64(self):
                raise ValueError("not base64")

            images = None
            usage = None

        out = _extract_xai_image(("prompt", "grok-imagine-image"), {}, _Img())
        self.assertEqual(out["items"]["image_size"], "")
        self.assertEqual(out["items"]["images_generated"], 1)

    def test_binary_from_base64(self):
        class _Img:
            base64 = PNG_2X3_B64
            images = None
            usage = None

        out = _extract_xai_image(("prompt", "grok-imagine-image"), {}, _Img())
        self.assertEqual(out["items"]["image_size"], "2x3")


class TestFailOpen(unittest.TestCase):
    def test_extractors_never_throw_on_garbage(self):
        for fn in (
            _extract_together_image,
            _extract_hf_image,
            _extract_google_imagen,
            _extract_openai_images,
            _extract_xai_image,
        ):
            try:
                fn((), {}, None)
                fn((), {"parameters": "not-a-dict", "size": object()}, object())
                fn((None,), {"width": "x"}, {"data": "bad"})
            except Exception as e:  # pragma: no cover
                self.fail(f"{fn.__name__} raised {e!r}")

    def test_binary_helpers_never_throw(self):
        for bad in (None, b"", b"\x00\x01", "not-bytes", object(), memoryview(b"x")):
            try:
                _image_dims_from_binary(bad)  # type: ignore[arg-type]
                _resolve_image_size(result=bad, provider="together")
            except Exception as e:  # pragma: no cover
                self.fail(f"helper raised {e!r} on {bad!r}")


if __name__ == "__main__":
    unittest.main()
