"""Importable helper for tests/test_protect.py::test_string_path_unchanged.

This module exists ONLY to give the string-path form of ``protect()`` a class
that is genuinely resolvable by dotted path (`importlib.import_module` +
`getattr`). Every OTHER protected class in test_protect.py is defined in
FUNCTION-LOCAL scope so it is non-importable (the premise); this one is
deliberately module-level so the legacy import path can find it.
"""


class StringHelperClient:
    def create(self, *args, **kwargs):
        return {"ok": True}
