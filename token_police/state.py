"""
Module-level singleton state for the SDK.

In addition to the client singleton (legacy), this module also holds the
v2 daemon-mode Decision Pack cache: the directive list, the monotonic
version counter, plus an observations queue with per-call attribution:
each entry is tagged (at push time) with the "obs key" minted by the
pre-flight check of the LLM call that produced it, and a /log drain claims
only its own call's entries — plus untagged entries (no key was current at
push) and stale orphans (whose owning /log never fired). Legacy no-arg
drains still return everything.

The SDK is called from both async and threaded contexts (the SSE reader
thread updates the pack while the enforcer reads it), so mutable state is
guarded by dedicated locks: `_instance_lock` serializes the client swap in
`set_client` (close-old → install-new → reset-pack is atomic w.r.t. a
concurrent re-init), while `_pack_lock`/`_observations_lock` guard the
Decision Pack cache and observations queue. `get_client()` is deliberately
NOT locked — it is the lock-free hot read path (a single-attribute read is
atomic under the GIL).
"""
from __future__ import annotations

from typing import Optional, Any, Dict, List, TYPE_CHECKING
import contextvars
import logging
import threading
import time
import uuid

if TYPE_CHECKING:
    from .client import TokenPolice

logger = logging.getLogger("token_police")


# ── Legacy: client singleton ──────────────────────────────────────────
_instance: Optional["TokenPolice"] = None
# Serializes the whole close-old → install-new → reset-pack swap in
# set_client so a concurrent re-init() on another thread can't tear down a
# client mid-swap. get_client() intentionally does NOT take this lock.
_instance_lock = threading.Lock()


def get_client() -> Optional["TokenPolice"]:
    return _instance


def set_client(client: "TokenPolice") -> None:
    global _instance, _client_id
    # Hold _instance_lock across the ENTIRE swap: without it a second init()
    # on thread B could run the old client's close_sync() (executor shutdown +
    # sync_client.close()) while thread A is mid close→swap, leaving _instance
    # transiently pointing at a half-closed client.
    with _instance_lock:
        if _instance and hasattr(_instance, "close_sync"):
            try:
                _instance.close_sync()
            except Exception:
                pass
        _instance = client
        # The module-global client id tracks the INSTALLED client only —
        # constructing a TokenPolice() without installing it must not touch it.
        cid = getattr(client, "client_id", None)
        if isinstance(cid, str) and cid:
            _client_id = cid
        # Reset the pack cache when the client changes so a re-init() in tests
        # doesn't carry stale state from the prior run.
        reset_pack()


# ── v2: Decision Pack cache ───────────────────────────────────────────
_pack: Optional[Dict[str, Any]] = None         # last fully-applied snapshot
_pack_version: int = 0                          # monotonic version
_pack_received_at: float = 0.0                  # unix-ts of last update
# Snapshot TTL in seconds, as stamped by the server in `ttl_seconds`. 0 /
# absent / non-positive ⇒ never expires (legacy-snapshot path stays
# byte-identical). Read ONLY by the expiry helpers below.
_pack_ttl_seconds: int = 0
_cache_invalid: bool = False                    # set on apply error / TTL exhaustion
_pack_lock = threading.Lock()
_observations_queue: List[Dict[str, Any]] = []
_observations_lock = threading.Lock()
_client_id: str = ""                            # installed client's id (set by set_client)
_pack_tenant_id: Optional[str] = None           # learned from first snapshot
_pack_project_id: Optional[str] = None


def _attach_derived_unlocked(pack: Optional[Dict[str, Any]]) -> None:
    """Build the hot-path structures the local evaluator precomputes (loop-block
    set, pre-sorted directives, per-directive entity frozensets) and stash them
    ON the pack under the private derived key, IN PLACE. Rebuilt from scratch on
    EVERY pack swap (snapshot + delta) so the derived data can never lag the pack
    it describes. Fully guarded: any precompute failure clears the key and leaves
    the pack usable — the evaluator then rebuilds the structures on the fly
    (identical results) — so the precompute can NEVER poison pack apply. MUST be
    called while holding _pack_lock (it mutates the live pack)."""
    if not isinstance(pack, dict):
        return
    try:
        from .local_evaluator import build_derived, _DERIVED_KEY
    except Exception:
        return  # can't import the evaluator — leave the pack un-derived
    derived = None
    try:
        derived = build_derived(pack)
    except Exception:
        derived = None
    if derived is not None:
        pack[_DERIVED_KEY] = derived
    else:
        # Clear any STALE derived carried over by dict(_pack) in the delta path,
        # so a failed rebuild never leaves the evaluator reading old structures.
        pack.pop(_DERIVED_KEY, None)


def reset_pack() -> None:
    global _pack, _pack_version, _pack_received_at, _pack_ttl_seconds, _cache_invalid
    global _observations_queue, _pack_tenant_id, _pack_project_id
    with _pack_lock:
        _pack = None
        _pack_version = 0
        _pack_received_at = 0.0
        _pack_ttl_seconds = 0
        _cache_invalid = False
        _pack_tenant_id = None
        _pack_project_id = None
    with _observations_lock:
        _observations_queue = []
    # Clear the CURRENT context's per-call obs key too (re-init hygiene: a
    # stale key must not tag pushes made after a client swap). Only affects
    # the calling context — other tasks/threads keep their own copies.
    try:
        _obs_call_key.set(None)
    except Exception:
        pass


def _pack_expired_unlocked() -> bool:
    """Pure, throw-free, LOCK-FREE receipt-based expiry predicate.

    MUST be called under an already-held ``_pack_lock`` (or with none held).
    The non-reentrant ``_pack_lock`` means the lock-holding accessors below
    inline THIS helper — never the lock-acquiring public
    ``is_pack_expired()`` — or they would deadlock. GATE: ttl<=0/absent ⇒ never
    expires (legacy path unchanged). Seconds throughout (``time.time()`` is
    seconds, so no ×1000 unlike Node's ms clock).
    """
    return (
        _pack is not None
        and _pack_ttl_seconds > 0
        and (time.time() - _pack_received_at) > _pack_ttl_seconds
    )


def is_pack_expired() -> bool:
    """Public expiry check (used by the SSE watchdog). Acquires _pack_lock once
    and delegates to the lock-free helper. NEVER call this from an accessor that
    already holds _pack_lock (non-reentrant deadlock) — use
    _pack_expired_unlocked() inline instead."""
    with _pack_lock:
        return _pack_expired_unlocked()


def get_pack() -> Optional[Dict[str, Any]]:
    """Return the current pack, or None if cache is invalid/unset/expired."""
    with _pack_lock:
        # Inline the LOCK-FREE helper under the lock we already hold; a call
        # to is_pack_expired() here would re-acquire _pack_lock → deadlock.
        if _cache_invalid or _pack is None or _pack_expired_unlocked():
            return None
        return _pack


def get_pack_version() -> int:
    with _pack_lock:
        return _pack_version


def get_client_id() -> str:
    """The installed (global) client's id, or "" before any client is installed.
    Assigned by set_client — constructing a client without installing it does
    not change this value."""
    return _client_id


def apply_snapshot(snapshot: Dict[str, Any]) -> bool:
    """
    Replace the cache with a fresh snapshot. Returns True on success.

    Multi-tenancy guard: a snapshot whose tenant_id/project_id doesn't match
    what we've previously learned is dropped and the cache is invalidated:
    rather than ever evaluate against the wrong tenant, the SDK falls back to
    per-call checks.
    """
    global _pack, _pack_version, _pack_received_at, _pack_ttl_seconds, _cache_invalid
    global _pack_tenant_id, _pack_project_id
    try:
        version = int(snapshot.get("version", 0))
        t = snapshot.get("tenant_id")
        p = snapshot.get("project_id")
        with _pack_lock:
            # A snapshot without a well-formed tenant/project identity is refused
            # and the cache is poisoned (→ inline /check) rather than cached — so
            # the cross-tenant pin only ever arms from a snapshot that actually
            # carries both ids. Without this, a first snapshot missing an id would
            # either leave the pin disarmed forever or arm it half-formed and
            # reject every later good snapshot. Covered by
            # test_apply_snapshot_missing_ids_refused_and_arms_pin_after_heal.
            if not (isinstance(t, str) and t and isinstance(p, str) and p):
                logger.warning(
                    "TokenPolice: snapshot missing tenant/project identity (got %r/%r) — poisoning cache",
                    t, p,
                )
                _cache_invalid = True
                return False
            if _pack_tenant_id is not None and (_pack_tenant_id != t or _pack_project_id != p):
                logger.warning(
                    "TokenPolice: snapshot tenant/project mismatch (got %s/%s, expected %s/%s) — poisoning cache",
                    t, p, _pack_tenant_id, _pack_project_id,
                )
                _cache_invalid = True
                return False
            _pack = snapshot
            _pack_version = version
            _pack_received_at = time.time()
            # Stamp the server TTL; non-numeric/absent ⇒ 0 ⇒ never expires.
            try:
                _pack_ttl_seconds = int(snapshot.get("ttl_seconds", 0) or 0)
            except (TypeError, ValueError):
                _pack_ttl_seconds = 0
            _cache_invalid = False
            _pack_tenant_id = t
            _pack_project_id = p
            # Precompute the evaluator's hot-path structures atomically with the
            # swap so the very first eval against this pack is already fast.
            _attach_derived_unlocked(_pack)
        return True
    except Exception as err:
        logger.debug("TokenPolice: apply_snapshot failed: %s", err)
        with _pack_lock:
            _cache_invalid = True
        return False


def apply_deltas(ops: List[Dict[str, Any]], new_version: int) -> bool:
    """
    Apply a contiguous delta. Returns True on success.

    Enforces strict version contiguity (new_version == _pack_version + 1).
    Any version <= _pack_version is a duplicate and silently discarded
    (returns True). Any gap poisons the cache and returns False — the SSE
    client will reconnect with Last-Event-ID to recover.
    """
    # `_pack_received_at` MUST be in this global list — the receipt refresh
    # below is a rebinding assignment; without it, that write would create a
    # silent function-LOCAL and the delta would never reset the expiry clock.
    global _pack, _pack_version, _pack_received_at, _cache_invalid
    try:
        with _pack_lock:
            if _pack is None or _cache_invalid:
                return False
            if new_version <= _pack_version:
                return True  # duplicate/late, idempotent discard
            if new_version != _pack_version + 1:
                logger.debug(
                    "TokenPolice: delta version gap (have v%d, got v%d) — poisoning cache",
                    _pack_version, new_version,
                )
                _cache_invalid = True
                return False

            directives = list(_pack.get("directives", []))
            loop_blocks = list(_pack.get("loop_blocks", []))
            dir_by_id = {d.get("id"): i for i, d in enumerate(directives)}

            for op in ops:
                kind = op.get("op")
                if kind == "directive_upserted":
                    d = op.get("directive") or {}
                    rid = d.get("id")
                    if not rid:
                        continue
                    if rid in dir_by_id:
                        prev = directives[dir_by_id[rid]]
                        # An upsert op may arrive without the runtime
                        # `entities` list; a full replace would transiently
                        # disarm already-armed entities on every rule edit, so
                        # carry the armed set forward when the rule kind is
                        # unchanged and entity-bearing (block vs reroute
                        # entities are different sets). The directive's own
                        # `mode` still gates enforcement, so a dry_run rule is
                        # never armed.
                        if ("entities" not in d and isinstance(prev, dict)
                                and prev.get("kind") == d.get("kind")
                                and d.get("kind") in ("ENTITY_BLOCK", "REROUTE")
                                and isinstance(prev.get("entities"), list)):
                            d["entities"] = prev.get("entities")
                        directives[dir_by_id[rid]] = d
                    else:
                        dir_by_id[rid] = len(directives)
                        directives.append(d)
                elif kind == "directive_removed":
                    rid = op.get("rule_id")
                    if rid in dir_by_id:
                        idx = dir_by_id.pop(rid)
                        directives.pop(idx)
                        # rebuild index for shifted entries
                        dir_by_id = {d.get("id"): i for i, d in enumerate(directives)}
                    elif rid:
                        # Truthy rule_id that misses the index means the
                        # removal op referenced a directive we don't have. Rather
                        # than silently advance the version (leaving a removed rule
                        # enforcing until the next full snapshot), poison the cache
                        # and bail so the SSE caller re-snapshots. Set the flag
                        # INLINE — we already hold the non-reentrant _pack_lock, so
                        # the invalidate_pack helper must NOT be used here (it
                        # re-acquires the same lock and would deadlock).
                        _cache_invalid = True
                        return False
                    # falsy rid: benign no-op, version still advances (unchanged).
                elif kind in ("entity_blocked", "entity_rerouted"):
                    rid = op.get("rule_id")
                    entity = op.get("entity")
                    if not rid or entity is None:
                        continue
                    if rid in dir_by_id:
                        # Clone-before-mutate (COW). The directive dict is
                        # still the SAME object the live _pack points at; a
                        # concurrent daemon-mode local eval reads it AFTER the
                        # pack-read helper released _pack_lock. Mutating it in
                        # place would let that in-flight reader observe a
                        # half-armed entity list. Copy the dict inline (dict(...)
                        # takes NO lock — the non-reentrant _pack_lock deadlock
                        # hazard: never call a lock-reacquiring pack helper
                        # here) and write the clone back so the reader's
                        # captured dict is never touched.
                        d = dict(directives[dir_by_id[rid]])
                        ents = list(d.get("entities") or [])
                        if entity not in ents:
                            ents.append(entity)
                        d["entities"] = ents
                        directives[dir_by_id[rid]] = d
                elif kind in ("entity_unblocked", "entity_unrerouted"):
                    rid = op.get("rule_id")
                    entity = op.get("entity")
                    if not rid or entity is None:
                        continue
                    if rid in dir_by_id:
                        # Clone-before-mutate (COW) — same reader-isolation
                        # reasoning as the arm branch above. Inline dict(...) only
                        # (no _pack_lock-reacquiring helper), then write back.
                        d = dict(directives[dir_by_id[rid]])
                        ents = [e for e in (d.get("entities") or []) if e != entity]
                        d["entities"] = ents
                        directives[dir_by_id[rid]] = d
                elif kind == "loop_blocked":
                    tid = op.get("trace_id")
                    if tid and tid not in loop_blocks:
                        loop_blocks.append(tid)
                elif kind == "loop_unblocked":
                    tid = op.get("trace_id")
                    if tid:
                        loop_blocks = [x for x in loop_blocks if x != tid]
                else:
                    # Forward-compat: unknown op skipped, version advances.
                    logger.debug("TokenPolice: unknown delta op %r — skipped", kind)

            new_pack = dict(_pack)
            new_pack["directives"] = directives
            new_pack["loop_blocks"] = loop_blocks
            new_pack["version"] = new_version
            # Rebuild the precomputed structures for the NEW directive/loop state
            # (this also overwrites the stale derived key copied in by dict(_pack)
            # above, so the evaluator never reads structures from the old pack).
            _attach_derived_unlocked(new_pack)
            _pack = new_pack
            _pack_version = new_version
            # A successful delta is a fresh "update" — reset the receipt
            # clock so a pack kept current by a steady delta stream never falsely
            # expires (TTL = "no snapshot OR delta for ttl_seconds").
            _pack_received_at = time.time()
        return True
    except Exception as err:
        logger.debug("TokenPolice: apply_deltas threw: %s", err)
        with _pack_lock:
            _cache_invalid = True
        return False


def invalidate_pack(reason: str = "unspecified") -> None:
    """Mark the cache invalid; next reader will get None → inline /check."""
    global _cache_invalid
    with _pack_lock:
        _cache_invalid = True
    logger.debug("TokenPolice: pack invalidated (%s)", reason)


def is_cache_healthy() -> bool:
    with _pack_lock:
        # Inline the LOCK-FREE expiry helper under the held lock (never the
        # lock-acquiring is_pack_expired() — non-reentrant deadlock).
        return _pack is not None and not _cache_invalid and not _pack_expired_unlocked()


# ── SSE stream connection freshness ───────────────────────────────────
# Tracked SEPARATELY from pack health on purpose: a stream disconnect must
# never invalidate the pack — last-known-good blocking during an outage is
# design intent. These two fields only let the enforcer decide that a locally
# ALLOWED call whose decision hinged on a streamed entity list has gone too
# stale to trust (entity arming reaches the SDK over SSE and nowhere else), and
# re-verify it with the inline /check. The SSE reader thread writes them while
# enforcer threads read them, so they get their OWN lock — never nested with
# _pack_lock in either direction. Kept in parity with the Node SDK.
_stream_connected: bool = False
_stream_disconnected_at: Optional[float] = None
# Monotonic connection generation. These fields are module-global but every
# connection is owned by ONE reader thread, and a replaced client's reader can
# outlive the swap by up to the heartbeat window (it exits only when its parked
# iter_lines() unblocks). Without an owner check, that zombie's exit path would
# mark the LIVE stream of the NEW reader disconnected — permanently, since the
# new reader never re-marks connected. Each connect takes the next generation
# and only its owner may retire it.
_stream_generation: int = 0
_stream_lock = threading.Lock()


def mark_stream_connected() -> int:
    """Stream established (HTTP 200, body open). Returns the generation the
    caller must hand back to mark_stream_disconnected() when it ends."""
    global _stream_connected, _stream_disconnected_at, _stream_generation
    with _stream_lock:
        _stream_generation += 1
        _stream_connected = True
        _stream_disconnected_at = None
        return _stream_generation


def mark_stream_disconnected(generation: int) -> None:
    """Stream lost/closed, retiring ``generation``.

    A stale generation (a reader whose connection was already superseded) is
    ignored, so a late-exiting zombie can never mark a live stream down.
    IDEMPOTENT within a generation by design: the grace window is measured from
    the EARLIEST disconnect since the last successful connect, so the repeated
    calls a failing reconnect ladder produces must NOT refresh the stamp (that
    would keep extending the stale-allow window through a sustained outage).
    """
    global _stream_connected, _stream_disconnected_at
    with _stream_lock:
        if generation != _stream_generation:
            return
        if not _stream_connected:
            return
        _stream_connected = False
        _stream_disconnected_at = time.time()


def is_stream_fresh(grace_seconds: float) -> bool:
    """True while streamed entity lists can still be trusted: connected, or
    disconnected less than ``grace_seconds`` ago. Never-connected ⇒ False
    (defensive — the pack is None then, so the enforcer is already on the
    inline /check path)."""
    with _stream_lock:
        if _stream_connected:
            return True
        if _stream_disconnected_at is None:
            return False
        try:
            grace = float(grace_seconds)
        except (TypeError, ValueError):
            grace = 0.0
        if not grace > 0:  # also catches NaN
            grace = 0.0
        return (time.time() - _stream_disconnected_at) <= grace


# ── Observations queue ────────────────────────────────────────────────

# Staleness window for tagged-but-unclaimed observations. The window must
# exceed the longest plausible LLM call duration — a short window would let a
# concurrent /log re-steal a slow/streaming call's observations, recreating
# the cross-trace-attribution bug the per-call tagging fixes. A stale entry
# is an orphan whose owning /log never fired (crashed wrapper, abandoned
# stream); shipping it late on an arbitrary /log is deliberate — loss is
# strictly worse than late/misattributed delivery.
OBS_STALE_SECONDS = 300.0

# Per-call obs key. `_run_sync_check`/`_run_async_check` mint + set a fresh
# key at entry; pushes during check execution tag their entries with it, and
# wrappers capture it into a local right after their check to key the later
# drain. Concurrent asyncio Tasks copy their context at creation (no
# cross-task pollution) and threads have separate contexts; an inline awaited
# call SHARES the caller's context, which is exactly what lets the wrapper
# frame read the key its check just set.
_obs_call_key: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "tp_obs_call_key", default=None
)

# No-arg drain sentinel — distinguishes the legacy `drain_observations()`
# (drain ALL: shutdown paths, tests) from an explicit
# `drain_observations(claim_key=None)` (unknown claimant: untagged + stale
# only). A dedicated object, NOT None, so both spellings stay expressible.
_DRAIN_ALL = object()


def mint_obs_key() -> Optional[str]:
    """Mint a fresh per-call obs key and make it current. Fail-open: any
    error leaves the var untouched and returns the best-known value."""
    try:
        key = uuid.uuid4().hex
        _obs_call_key.set(key)
        return key
    except Exception:
        try:
            return _obs_call_key.get(None)
        except Exception:
            return None


def get_current_obs_key() -> Optional[str]:
    """The current call's obs key, or None outside any check context."""
    try:
        return _obs_call_key.get(None)
    except Exception:
        return None


def push_observation(obs: Dict[str, Any]) -> None:
    # PUBLIC SIGNATURE UNCHANGED — the entry is internally tagged with the
    # current call's obs key (None when no key is current / any read fails:
    # fail-open to the untagged pool, which every drain claims).
    if not isinstance(obs, dict):
        return
    try:
        key = _obs_call_key.get(None)
    except Exception:
        key = None
    try:
        ts = time.monotonic()
    except Exception:
        ts = 0.0
    with _observations_lock:
        _observations_queue.append({"obs": obs, "key": key, "ts": ts})


def drain_observations(claim_key: Any = _DRAIN_ALL) -> List[Dict[str, Any]]:
    """Drain observations for a /log POST, returning RAW observation dicts
    (the internal {"obs","key","ts"} wrapper never reaches a payload).

    - No argument → drain EVERYTHING (exact legacy semantics — shutdown
      callers and existing tests rely on this).
    - Explicit ``None`` (unknown claimant) → drain untagged entries + stale
      orphans (age > OBS_STALE_SECONDS) only.
    - A string key → drain that call's entries + untagged + stale; everything
      else stays queued for its own call's drain (order preserved on both
      the returned and remaining sides).
    """
    with _observations_lock:
        if claim_key is _DRAIN_ALL:
            out = [e["obs"] for e in _observations_queue]
            _observations_queue[:] = []
            return out
        now = time.monotonic()
        claimed: List[Dict[str, Any]] = []
        kept: List[Dict[str, Any]] = []
        for e in _observations_queue:
            if (
                e.get("key") is None
                or (isinstance(claim_key, str) and e.get("key") == claim_key)
                or (now - e.get("ts", 0.0)) > OBS_STALE_SECONDS
            ):
                claimed.append(e["obs"])
            else:
                kept.append(e)
        _observations_queue[:] = kept
        return claimed


def _observations_set(value: List[Dict[str, Any]]) -> None:
    """Internal helper for tests/atomics. Accepts RAW observation dicts and
    queues them UNTAGGED (claimed by any drain) — preserving the seam's
    pre-keying semantics for existing tests. Also clears the current-context
    obs key so a later manual push in the SAME test context goes untagged
    instead of inheriting a stale key from an earlier check."""
    global _observations_queue
    try:
        _obs_call_key.set(None)
    except Exception:
        pass
    with _observations_lock:
        _observations_queue = [
            {"obs": v, "key": None, "ts": time.monotonic()} for v in value
        ]


def _observations_set_entries(entries: List[Dict[str, Any]]) -> None:
    """Internal test seam: replace the queue with explicit
    {"obs","key","ts"} entries so tests can inject keys and ages."""
    global _observations_queue
    with _observations_lock:
        _observations_queue = [
            {
                "obs": e.get("obs"),
                "key": e.get("key"),
                "ts": e.get("ts", time.monotonic()),
            }
            for e in entries
        ]
