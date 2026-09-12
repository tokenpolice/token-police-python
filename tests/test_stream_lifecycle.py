"""StreamClient start()/stop() lifecycle.

stop() must not permanently kill the reader: a stop() followed by an immediate
start() spawns a live reader again (per-run stop event captured at thread
creation). start() is lock-guarded so concurrent starts spawn exactly one live
reader. Neither start() nor stop() ever join() — a customer thread must not
block on ours.
"""
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from token_police.stream import StreamClient


def _wait_until(pred, timeout=5.0, interval=0.01):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return pred()


def _make_client():
    return StreamClient(
        base_url="http://127.0.0.1:59999", api_key="tp_sk_test", sdk_version="1",
        deployment="daemon", client_id="c1",
    )


def test_stop_then_immediate_start_revives_reader():
    sc = _make_client()

    def fake_connect(stop_event):
        # Simulate a live connection: block until this reader's own event fires.
        stop_event.wait(30)

    sc._connect_and_pump = fake_connect
    try:
        sc.start()
        assert _wait_until(lambda: sc._thread is not None and sc._thread.is_alive())
        first = sc._thread

        sc.stop()
        assert _wait_until(lambda: not first.is_alive())

        # Immediately restart — the previously-set stop event must NOT prevent a
        # fresh reader from running.
        sc.start()
        assert _wait_until(lambda: sc._thread is not None and sc._thread.is_alive())
        assert sc._thread is not first
    finally:
        sc.stop()


def test_concurrent_start_spawns_exactly_one_reader():
    sc = _make_client()
    seen = set()
    seen_lock = threading.Lock()

    def fake_connect(stop_event):
        with seen_lock:
            seen.add(threading.current_thread())
        stop_event.wait(30)

    sc._connect_and_pump = fake_connect
    try:
        barrier = threading.Barrier(8)

        def caller():
            barrier.wait()
            sc.start()

        callers = [threading.Thread(target=caller) for _ in range(8)]
        for c in callers:
            c.start()
        for c in callers:
            c.join()

        # Wait for the (single) reader to actually enter the fake connection.
        assert _wait_until(lambda: len(seen) >= 1)
        # Give any spurious extra readers a chance to appear, then assert none did.
        time.sleep(0.1)
        with seen_lock:
            alive = [t for t in seen if t.is_alive()]
        assert len(alive) == 1, f"expected exactly one live reader, got {len(alive)}"
    finally:
        sc.stop()


def test_stop_does_not_block():
    # stop() must return promptly (never join the reader).
    sc = _make_client()

    def fake_connect(stop_event):
        stop_event.wait(30)

    sc._connect_and_pump = fake_connect
    try:
        sc.start()
        assert _wait_until(lambda: sc._thread is not None and sc._thread.is_alive())
        t0 = time.time()
        sc.stop()
        assert time.time() - t0 < 1.0
    finally:
        sc.stop()
