"""
SSE client for /v1/guard/stream.

Runs on a daemon thread, maintains a long-lived HTTP/1.1 streaming
connection, parses SSE frames, and pumps snapshots + deltas into
state.apply_snapshot / state.apply_deltas. On connection loss reconnects
with exponential backoff capped at sse_reconnect_max_interval_seconds
(default 300 s). On gap detection or apply failure reconnects without
Last-Event-ID to force a fresh snapshot.

This is daemon-mode only (firewall 'enforce' or 'dry_run') — serverless and
edge can't keep a socket open, and 'off' mode doesn't need the cache. Caller
decides when to start the reader.
"""
from __future__ import annotations

import httpx
import json
import logging
import random
import threading
import time
from typing import Optional

from . import state

logger = logging.getLogger("token_police")

HEARTBEAT_TIMEOUT_SECONDS = 45    # 3 missed 15s keepalives → force reconnect
STALE_DATA_THRESHOLD_SECONDS = 600  # no data event in 10 min → reconnect


class StreamClient:
    """Wraps a daemon thread that maintains the SSE connection."""

    def __init__(self, base_url: str, api_key: str, sdk_version: str,
                 deployment: str, client_id: str,
                 firewall: str = "dry_run",
                 reconnect_cap_seconds: int = 300):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.sdk_version = sdk_version
        self.deployment = deployment
        self.client_id = client_id
        self.firewall = firewall
        self.reconnect_cap_seconds = max(60, int(reconnect_cap_seconds or 300))
        self._thread: Optional[threading.Thread] = None
        # Guards start()/stop() against concurrent restarts. _stop_event is the
        # CURRENT reader's event; each reader thread captures its own event at
        # creation, so signalling this one only stops the reader that owns it.
        self._start_lock = threading.Lock()
        self._stop_event = threading.Event()
        # Liveness generation owned by the CURRENT connection (0 = none yet).
        # Set at connect, handed back on exit so a superseded reader can't
        # retire a live stream. Stays set after an exit: a reconnect attempt
        # that fails before connecting re-marks the same (already-retired)
        # generation, which no-ops.
        self._stream_generation = 0
        self._last_event_at = 0.0
        self._last_data_event_at = 0.0
        # Reconnect cursor — None forces a fresh snapshot.
        self._last_event_id: Optional[int] = None
        # Per-connection "force immediate reconnect" flag. Set only when a
        # delta arrives while the cache is unhealthy (and by the snapshot-TTL
        # watchdog); consumed by the loop-top guard in _pump so exiting closes
        # the socket and the existing backoff reconnects WITHOUT
        # Last-Event-ID (→ fresh snapshot). Reset per-connection at the top of
        # _pump so a flag set on connection N can never kill connection N+1.
        # Never sets _stop_event.
        self._force_reconnect = False
        # A terminal connect status (401/403/404) does NOT stop the reader. It
        # records the status here so the outer loop re-probes at the reconnect
        # cap (ladder top) instead of the ~1s floor; cleared once consumed.
        self._terminal_status: Optional[int] = None
        # True while consecutive connects keep returning a terminal status, so
        # the streak is logged at warning once and at debug thereafter.
        self._in_terminal_streak = False
        # Per-connection "traffic evidence" flag: True once the current
        # connection has delivered any dispatched frame or any comment other
        # than the `: stream-open` banner. Gates the attempt=0 reset in
        # _run_forever — a 200 that EOFs before any traffic is a failed
        # connect in disguise, and resetting backoff for it collapses the
        # ladder into a ~1 Hz reconnect storm. stream-open doesn't count as
        # traffic: the server writes it before subscribe/snapshot on every
        # connection, including ones about to fail, so it proves nothing.
        self._saw_stream_traffic = False

    # ── Lifecycle ─────────────────────────────────────────────────────
    def start(self) -> None:
        with self._start_lock:
            current = self._thread
            # Only no-op if the current reader is alive AND not being stopped.
            # A thread that is dead or already signalled to stop can never
            # recover (its event stays set), so spawn a fresh one.
            if current and current.is_alive() and not self._stop_event.is_set():
                return
            # Signal the old reader (its OWN captured event) to exit, then hand
            # the new reader a fresh event. Never join() — a customer thread must
            # never block on ours. A brief overlap of the old and new readers is
            # harmless: snapshot/delta apply is idempotent.
            self._stop_event.set()
            stop_event = threading.Event()
            self._stop_event = stop_event
            self._thread = threading.Thread(
                target=self._run_forever, args=(stop_event,),
                name="tp-stream", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        # Signal the current reader's event. A start() racing this may install a
        # fresh reader; the loser's event is simply discarded.
        # Only signals. Retiring the connection is left to the reader's own exit
        # path below, which owns the generation — stop() marking here would have
        # to guess one, and a wrong guess is exactly the zombie-marks-live-stream
        # bug the generation check exists to prevent.
        with self._start_lock:
            self._stop_event.set()

    # ── Reader ────────────────────────────────────────────────────────
    def _run_forever(self, stop_event: Optional[threading.Event] = None) -> None:
        # start() always passes the reader's OWN captured event; the default is
        # only for direct callers (tests) and binds to the current event.
        if stop_event is None:
            stop_event = self._stop_event
        backoff_steps = [1, 2, 4, 8, 16, 32, 64, 128, 256, self.reconnect_cap_seconds]
        attempt = 0
        while not stop_event.is_set():
            try:
                self._connect_and_pump(stop_event)
                if self._terminal_status is not None:
                    # The last connect returned a terminal status (401/403/404)
                    # without stopping. Re-probe at the reconnect cap (ladder
                    # top) rather than the ~1s floor — a persistently terminal
                    # endpoint is polled at most once per cap period, never
                    # hammered. A later non-terminal connect resets attempt=0.
                    self._terminal_status = None
                    attempt = len(backoff_steps) - 1
                else:
                    # Reset backoff only when the disconnect showed traffic
                    # evidence (a frame or a keepalive). A clean 200 EOF with
                    # no traffic ever is a failed connect in disguise —
                    # preserve `attempt` so the ladder climbs (1→2→4→8…) like
                    # the transient path, instead of storming at the ~1s floor.
                    if self._saw_stream_traffic:
                        attempt = 0
            except Exception as err:
                logger.debug("TokenPolice: stream error: %s", err)
            finally:
                # EVERY pump exit is a disconnect: transient raise, heartbeat /
                # data-stale watchdog return, clean EOF, forced-reconnect +
                # snapshot-TTL branch, terminal status, stop. Retires only THIS
                # reader's generation (0 = never connected, which no generation
                # ever equals), so a superseded reader exiting late cannot mark
                # the replacement's live stream down. Idempotent +
                # earliest-stamp within a generation (see state.py), so a
                # failing reconnect ladder can never refresh the staleness
                # clock. Pack state is deliberately untouched — a disconnect
                # never invalidates it. Guarded: a raise in this finally would
                # kill the reader thread outright.
                try:
                    state.mark_stream_disconnected(self._stream_generation)
                except Exception:
                    pass
            if stop_event.is_set():
                return
            # Backoff with ±20% jitter, capped.
            base = backoff_steps[min(attempt, len(backoff_steps) - 1)]
            jitter = base * 0.2 * (random.random() * 2 - 1)
            sleep_for = min(self.reconnect_cap_seconds, max(1.0, base + jitter))
            attempt += 1
            stop_event.wait(sleep_for)

    def _connect_and_pump(self, stop_event: Optional[threading.Event] = None) -> None:
        if stop_event is None:
            stop_event = self._stop_event
        # httpx can set request headers (unlike browser EventSource), so the
        # api key rides Authorization: Bearer rather than the query string —
        # keeps it out of proxy/access logs. The server looks the key up by
        # hash, never plaintext.
        url = f"{self.base_url}/v1/guard/stream"
        headers = {
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
            "Authorization": f"Bearer {self.api_key}",
            "X-TP-Sdk-Version": self.sdk_version,
            "X-TP-Sdk-Schema-Version": "1",
            "X-TP-Client-Id": self.client_id,
            "X-TP-Deployment-Mode": self.deployment,
            "X-TP-Firewall-Mode": self.firewall,
        }
        if self._last_event_id is not None:
            headers["Last-Event-ID"] = str(self._last_event_id)

        # read timeout sits just above the heartbeat window. A healthy server
        # sends a `: keepalive` every 15s, so any 45s gap between bytes means
        # the connection went silent (dead-but-open socket); httpx then raises
        # ReadTimeout and _run_forever reconnects with backoff. Without this
        # (read=None) the in-loop watchdog below can never fire — iter_lines
        # blocks forever on a half-open socket and the reader hangs.
        timeout = httpx.Timeout(connect=10.0, read=float(HEARTBEAT_TIMEOUT_SECONDS),
                                write=10.0, pool=10.0)
        with httpx.stream("GET", url, headers=headers, timeout=timeout) as resp:
            if resp.status_code != 200:
                # A terminal connect status (revoked/invalid key → 401/403, or a
                # misconfigured base_url/route → 404) will not self-heal on a fast
                # retry. Rather than hammer the endpoint at the ~1s backoff floor
                # (a tight reconnect loop), invalidate the pack FIRST — the TTL
                # watchdog lives inside _pump, so leaving a still-healthy pack in
                # place would enforce a stale pack forever; unhealthy → enforcer
                # bails to inline /check (fail-open) — then mark this connect
                # terminal and return WITHOUT setting the stop signal. The outer
                # loop re-probes at the reconnect cap (~one probe per cap period);
                # a later non-terminal connect resumes normal streaming. stop()
                # still interrupts the cap-length wait instantly. A terminal
                # status is logged at warning once per streak, then at debug to
                # avoid spam.
                # Drop Last-Event-ID here too (every other poison path already
                # does). Collector skip-when-equal withholds the snapshot that
                # clears the poison; a leftover cursor on an idle project never
                # heals (SSE-12).
                if resp.status_code in (401, 403, 404):
                    state.invalidate_pack(f"stream_terminal_{resp.status_code}")
                    self._last_event_id = None
                    self._terminal_status = resp.status_code
                    if not self._in_terminal_streak:
                        logger.warning(
                            "TokenPolice: rule stream got HTTP %s; will re-probe in "
                            "~%ss (local rule evaluation degraded to inline checks "
                            "meanwhile)",
                            resp.status_code, self.reconnect_cap_seconds,
                        )
                    else:
                        logger.debug(
                            "TokenPolice: /stream connect %s (still terminal)",
                            resp.status_code,
                        )
                    self._in_terminal_streak = True
                    return
                # A TRANSIENT connect failure (429/5xx/other-4xx) must NOT
                # bare-return — a clean return resets _run_forever's attempt=0,
                # collapsing the exponential backoff ladder to its ~1s floor (a
                # reconnect storm on /v1/guard/stream during a sustained outage).
                # Raise instead so the existing _run_forever except preserves the
                # climbed `attempt` and the next reconnect climbs the ladder
                # (1→2→4→8…). A transient response also ends any terminal streak.
                self._in_terminal_streak = False
                logger.debug("TokenPolice: /stream connect %s", resp.status_code)
                raise RuntimeError(f"stream connect {resp.status_code}")
            now = time.time()
            # Connection actually established (200 + body) — every terminal /
            # transient status check above has passed. Capture the generation
            # this connection owns; only it may retire the liveness state (see
            # _run_forever's finally).
            self._stream_generation = state.mark_stream_connected()
            # No traffic evidence yet for this connection — reset here (not
            # inside _pump) so a 200 whose body dies before the pump ever
            # runs still counts as traffic-less and climbs the backoff ladder.
            self._saw_stream_traffic = False
            # A successful connect ends any terminal streak.
            self._in_terminal_streak = False
            self._last_event_at = now
            self._last_data_event_at = now
            self._pump(resp, stop_event)

    def _pump(self, response: httpx.Response,
              stop_event: Optional[threading.Event] = None) -> None:
        if stop_event is None:
            stop_event = self._stop_event
        # The real stall bound is the outer httpx read timeout (set to the
        # heartbeat window in _connect_and_pump): a silent half-open socket
        # raises ReadTimeout and _run_forever reconnects. The in-loop checks
        # below only fire when a line actually arrives.
        event_type = "message"
        event_id: Optional[str] = None
        data_lines: list[str] = []
        # Fresh connection starts with a clear reconnect flag.
        self._force_reconnect = False
        for line in response.iter_lines():
            if stop_event.is_set():
                return
            # An unhealthy-cache delta asked for an immediate reconnect.
            # Return (no raise) → the `with httpx.stream(...)` block closes the
            # socket, _connect_and_pump returns, _run_forever resets attempt=0
            # and the existing backoff reconnects with no Last-Event-ID so the
            # server sends a fresh snapshot.
            if self._force_reconnect:
                return
            now = time.time()
            # Watchdog: if heartbeats stop landing, force reconnect.
            if now - self._last_event_at > HEARTBEAT_TIMEOUT_SECONDS:
                logger.debug("TokenPolice: heartbeat timeout — reconnecting")
                return
            if now - self._last_data_event_at > STALE_DATA_THRESHOLD_SECONDS:
                logger.debug("TokenPolice: data-stale timeout — reconnecting")
                return
            # The pack aged past its server-stamped TTL while heartbeats kept
            # arriving but no snapshot/delta refreshed it. Invalidate
            # (→ inline /check) then reuse the same "force immediate
            # reconnect" flag: null the cursor and set _force_reconnect so the
            # loop-top guard in _pump returns on the next iteration → the
            # `with httpx.stream(...)` block closes and the existing backoff
            # reconnects WITHOUT Last-Event-ID → fresh snapshot → pack
            # un-expires. is_pack_expired() and invalidate_pack() are two
            # SEQUENTIAL (non-nested) _pack_lock acquisitions — never nest
            # them; the lock is non-reentrant and nesting deadlocks. No new
            # backoff/socket code.
            if state.is_pack_expired():
                logger.debug("TokenPolice: snapshot TTL expired — reconnecting")
                state.invalidate_pack("snapshot_ttl_expired")
                self._last_event_id = None
                self._force_reconnect = True
            self._last_event_at = now

            if line == "":
                # End of event frame.
                if data_lines:
                    data = "\n".join(data_lines)
                    # Traffic evidence at dispatch time regardless of whether
                    # apply succeeds — a well-formed frame proves the server
                    # pipeline is alive.
                    self._saw_stream_traffic = True
                    self._dispatch(event_type, event_id, data)
                    self._last_data_event_at = time.time()
                event_type, event_id, data_lines = "message", None, []
                continue
            if line.startswith(":"):
                # comment / keepalive — already updated _last_event_at above.
                # Any comment except the stream-open banner is traffic
                # evidence (the banner precedes subscribe/snapshot on every
                # connection, including ones about to fail).
                if line[1:].strip() != "stream-open":
                    self._saw_stream_traffic = True
                continue
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
            elif line.startswith("id:"):
                event_id = line[len("id:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].lstrip())
            # everything else is ignored per SSE spec

    # ── Frame handling ────────────────────────────────────────────────
    def _dispatch(self, event_type: str, event_id: Optional[str], data: str) -> None:
        try:
            payload = json.loads(data)
        except Exception:
            logger.debug("TokenPolice: malformed SSE payload, dropping")
            return

        if event_type == "snapshot":
            ok = state.apply_snapshot(payload)
            if ok and event_id and event_id.isdigit():
                self._last_event_id = int(event_id)
            elif not ok:
                # Apply FAILED → drop cache and cursor so the next reconnect
                # pulls a fresh snapshot. A successful apply that merely lacks a
                # numeric id keeps the pack healthy and leaves the cursor as-is
                # (mirrors the delta branch below) — a missing id must never
                # discard a snapshot that applied cleanly.
                state.invalidate_pack("snapshot_apply_failed")
                self._last_event_id = None
            return

        if event_type == "delta":
            version = int(payload.get("version", 0)) if isinstance(payload, dict) else 0
            ops = payload.get("ops") if isinstance(payload, dict) else []
            if not version or not isinstance(ops, list):
                return
            # Tenant guard — we know our tenant/project from the prior snapshot.
            if state.is_cache_healthy():
                ok = state.apply_deltas(ops, version)
                if ok and event_id and event_id.isdigit():
                    self._last_event_id = int(event_id)
                elif not ok:
                    # Gap / apply error — invalidate so we get a fresh snapshot
                    # on reconnect and don't advance Last-Event-ID.
                    self._last_event_id = None
                    state.invalidate_pack("delta_apply_failed")
            else:
                # Delta arrived while the cache is unhealthy (no snapshot
                # landed yet, or a prior apply/TTL failure poisoned it). Dropping
                # the cursor is inert without a reconnect (Last-Event-ID is only
                # sent in the connect headers), so also force an
                # immediate reconnect → the server pushes a fresh snapshot and
                # the cache self-heals.
                self._last_event_id = None
                self._force_reconnect = True
