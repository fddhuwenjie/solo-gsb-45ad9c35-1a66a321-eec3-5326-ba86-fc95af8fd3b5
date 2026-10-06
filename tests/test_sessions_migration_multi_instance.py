"""Targeted tests for upgrading an old (pre-wide-audit) sessions database when
two service instances share it.

The startup migration used to run as several autocommitted statements: check
the wide-column types, build a fixed-name staging table, copy, drop the
original and rename. Two instances starting at the same time interleaved
those statements, so startup could fail midway and a crash between DROP and
RENAME destroyed the historical sessions. The migration now runs as one
BEGIN IMMEDIATE write transaction:

* one instance upgrades; the other blocks on SQLite's write lock, waits for
  the upgrade to commit, then re-reads the structure and finishes startup
  without touching it (true cross-process version exercised with two
  processes; in-process tests pin the wait/re-check deterministically);
* any failure rolls the transaction back: the original INTEGER sessions
  table and every historical row survive, no staging table lingers, and the
  next startup completes the upgrade;
* a staging table orphaned by a crash of the *previous* (non-transactional)
  release is handled: rebuilt alongside the original table, or adopted when
  the original table is already gone;
* after the upgrade both instances query the historical session, replay its
  last shard and continue appending sequences past 2**63 with exact values.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import sqlite3
import threading

import pytest
from fastapi.testclient import TestClient

from app import db as dbmod
from app.config import Settings
from app.db import _connect, _initialize_schema
from app.main import create_app
from tests.conftest import b64
from tests.test_sequence_audit_wide import (
    FRAME64_LEN,
    INT64_MAX,
    _AUDIT_ERA_SESSIONS_DDL,
    frame64,
)


def create_audit_era_database(db_path, *, sessions_ddl=_AUDIT_ERA_SESSIONS_DDL):
    """Create a pre-wide-audit database file (wide columns still INTEGER)."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(sessions_ddl)
        conn.commit()
    finally:
        conn.close()


def insert_audit_era_session(db_path, session_id="old-session", last_sequence=INT64_MAX):
    """Insert one historical audit-era row carrying a replayable last shard."""
    chunk = frame64(last_sequence)
    stored_response = {
        "session_id": session_id,
        "protocol_id": "sensor-line64",
        "version": 1,
        "received_offset": 0,
        "next_offset": FRAME64_LEN,
        "frames": [],
        "discarded": [],
        "tail": {"start": FRAME64_LEN, "end": FRAME64_LEN, "payload_b64": ""},
        "frame_count": 0,
        "frames_total": 1,
        "replayed": False,
        "sequence_audit": {
            "bit_width": 64,
            "last_sequence": last_sequence,
            "anomalies": [],
            "duplicates": 1,
            "suspected_gaps": 0,
            "backwards": 0,
            "indeterminates": 0,
            "missing_total": 5,
        },
    }
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO sessions "
            "(session_id, protocol_id, version, received_offset, tail_b64, "
            " frames_total, last_chunk_offset, last_chunk_sha256, last_response, "
            " audit_enabled, audit_bit_width, audit_last_sequence, "
            " audit_duplicates, audit_missing_total) "
            "VALUES (?, 'sensor-line64', 1, ?, '', 1, 0, ?, ?, 1, 64, ?, 1, 5)",
            (
                session_id,
                FRAME64_LEN,
                hashlib.sha256(chunk).hexdigest(),
                json.dumps(stored_response),
                last_sequence,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return chunk


def _wide_column_types(db_path):
    probe = sqlite3.connect(db_path)
    try:
        return {
            row[1]: row[2].upper()
            for row in probe.execute("PRAGMA table_info(sessions)")
        }
    finally:
        probe.close()


def _table_names(db_path):
    probe = sqlite3.connect(db_path)
    try:
        return {
            row[0]
            for row in probe.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    finally:
        probe.close()


# --------------------------------- true cross-process simultaneous startup ----


_U64_PROTOCOL = {
    "protocol_id": "sensor-line64",
    "version": 1,
    "sync": [0xAA, 0x55],
    "endian": "big",
    "length_field": {"offset": 12, "size": 1, "adjustment": 0},
    "crc": {"preset": "CRC-16/CCITT-FALSE", "offset": 14},
    "fields": [
        {"name": "seq", "type": "u64", "offset": 0},
        {"name": "addr", "type": "u8", "offset": 8},
    ],
    "sequence_field": "seq",
    "max_frame_length": 64,
}


def _append(client, sid, offset, chunk):
    return client.post(
        "/sessions/append",
        json={
            "session_id": sid,
            "expected_offset": offset,
            "chunk_b64": b64(chunk),
        },
    )


def _concurrent_upgrade_worker(db_path, barrier, outcomes):
    """One of two service instances booting against the same old database."""
    try:
        # Both processes reach this point together, so the two
        # BEGIN IMMEDIATE startup transactions genuinely race.
        barrier.wait(20)
        from app.db import Database

        Database(db_path)  # runs the startup migration
        outcomes.put("ok")
    except Exception as exc:  # pragma: no cover - reported back for assertion
        outcomes.put(f"error: {exc!r}")


def test_two_processes_starting_together_upgrade_old_database(tmp_path):
    db_file = tmp_path / "audit-era.db"
    create_audit_era_database(str(db_file))
    chunk = insert_audit_era_session(str(db_file))

    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(2)
    outcomes = ctx.Queue()
    workers = [
        ctx.Process(
            target=_concurrent_upgrade_worker,
            args=(str(db_file), barrier, outcomes),
        )
        for _ in range(2)
    ]
    for w in workers:
        w.start()
    for w in workers:
        w.join(30)
        assert w.exitcode == 0, f"upgrade worker exited with {w.exitcode}"
    results = sorted(outcomes.get(timeout=5) for _ in range(2))
    assert results == ["ok", "ok"]

    # Exactly one TEXT rebuild happened: no staging table lingers and the
    # historical row kept its exact values as decimal text.
    assert dbmod._MIGRATION_STAGING_TABLE not in _table_names(str(db_file))
    types = _wide_column_types(str(db_file))
    assert types["audit_last_sequence"] == "TEXT"
    assert types["audit_missing_total"] == "TEXT"
    probe = sqlite3.connect(str(db_file))
    try:
        row = probe.execute(
            "SELECT audit_last_sequence, typeof(audit_last_sequence), "
            "       audit_missing_total, typeof(audit_missing_total) "
            "FROM sessions WHERE session_id = 'old-session'"
        ).fetchone()
        assert row == (str(INT64_MAX), "text", "5", "text")
    finally:
        probe.close()

    # Two independent app instances (each its own connection pool, like the
    # two processes above) share the upgraded file.
    settings = Settings(db_path=str(db_file), max_decode_bytes=1 << 20)
    client_a = TestClient(create_app(settings))
    client_b = TestClient(create_app(settings))
    with client_a, client_b:
        response = client_a.post("/protocols", json=_U64_PROTOCOL)
        assert response.status_code == 201, response.text

        # Both instances query the historical session with its exact
        # pre-upgrade audit values.
        for client in (client_a, client_b):
            progress = client.get("/sessions/old-session").json()
            assert progress["received_offset"] == FRAME64_LEN
            audit = progress["sequence_audit"]
            assert audit["last_sequence"] == INT64_MAX
            assert audit["duplicates"] == 1
            assert audit["missing_total"] == 5

        # Instance A replays the most recent shard; the stored response
        # comes back verbatim and counts nothing again.
        replay = _append(client_a, "old-session", 0, chunk).json()
        assert replay["replayed"] is True
        assert replay["sequence_audit"]["last_sequence"] == INT64_MAX
        assert replay["sequence_audit"]["missing_total"] == 5

        # Instance B continues the stream past the signed INTEGER boundary;
        # the migrated baseline 2**63-1 judges 2**63 as one normal step and
        # the value persists exactly.
        continued = _append(
            client_b, "old-session", FRAME64_LEN, frame64(1 << 63)
        ).json()
        assert continued["sequence_audit"]["anomalies"] == []
        assert continued["sequence_audit"]["last_sequence"] == 1 << 63

        # Both instances then observe B's continuation.
        for client in (client_a, client_b):
            audit = client.get("/sessions/old-session").json()["sequence_audit"]
            assert audit["last_sequence"] == 1 << 63
            assert audit["missing_total"] == 5


# ------------------------- wait for the peer, then re-judge the structure ----


def test_migration_blocks_on_peer_write_lock_then_rechecks_and_skips(db_path):
    create_audit_era_database(db_path)
    insert_audit_era_session(db_path)

    # A peer instance holds the startup write lock and performs the whole
    # upgrade itself while holding it.
    holder = _connect(db_path, autocommit=True)
    holder.execute("BEGIN IMMEDIATE")
    for statement in dbmod._SCHEMA:
        holder.execute(statement)
    dbmod._migrate_sessions(holder)

    # The late instance's startup migration is blocked on the same lock.
    late = _connect(db_path, autocommit=True)
    finished = threading.Event()
    failure: list[BaseException] = []

    def run_late_migration():
        try:
            _initialize_schema(late)
        except BaseException as exc:  # pragma: no cover - surfaced below
            failure.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=run_late_migration)
    thread.start()
    assert not finished.wait(0.5), "late instance ran while the peer held the lock"

    holder.execute("COMMIT")  # peer upgrade lands
    assert finished.wait(10)
    thread.join()
    assert failure == []

    # The late instance re-read the freshly migrated structure and did not
    # rebuild anything a second time.
    assert dbmod._MIGRATION_STAGING_TABLE not in _table_names(db_path)
    assert _wide_column_types(db_path)["audit_last_sequence"] == "TEXT"
    late.close()
    holder.close()


# ------------------------------- failure rolls back; restart retries --------


class _FailingAtSessionsDrop:
    """Connection wrapper that fails the old-table drop (after the copy)."""

    def __init__(self, inner):
        self._inner = inner

    def execute(self, sql, *args, **kwargs):
        if sql == "DROP TABLE sessions":
            raise sqlite3.OperationalError("injected failure at table swap")
        return self._inner.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_failed_migration_preserves_old_table_and_restart_completes_it(
    db_path, monkeypatch
):
    create_audit_era_database(db_path)
    chunk = insert_audit_era_session(db_path)

    real_connect = dbmod._connect
    monkeypatch.setattr(
        dbmod,
        "_connect",
        lambda path, **kw: _FailingAtSessionsDrop(real_connect(path, **kw)),
    )
    with pytest.raises(sqlite3.OperationalError, match="injected failure"):
        dbmod.init_db(db_path)
    monkeypatch.undo()

    # The transaction rolled back completely: the original INTEGER table and
    # its rows are intact, and no half-built staging table remains.
    assert _table_names(db_path) == {"sessions"}
    types = _wide_column_types(db_path)
    assert types["audit_last_sequence"] == "INTEGER"
    assert types["audit_missing_total"] == "INTEGER"
    probe = sqlite3.connect(db_path)
    try:
        row = probe.execute(
            "SELECT session_id, received_offset, audit_last_sequence, "
            "       audit_missing_total, frames_total FROM sessions"
        ).fetchone()
        assert row == ("old-session", FRAME64_LEN, INT64_MAX, 5, 1)
    finally:
        probe.close()

    # The failed attempt is not cached as "initialised"; the next startup
    # retries and completes the upgrade against the intact history.
    assert db_path not in dbmod._initialized
    dbmod.init_db(db_path)
    assert _wide_column_types(db_path)["audit_last_sequence"] == "TEXT"

    with TestClient(
        create_app(Settings(db_path=db_path, max_decode_bytes=1 << 20))
    ) as client:
        response = client.post("/protocols", json=_U64_PROTOCOL)
        assert response.status_code == 201, response.text
        audit = client.get("/sessions/old-session").json()["sequence_audit"]
        assert audit["last_sequence"] == INT64_MAX
        assert audit["missing_total"] == 5
        replay = _append(client, "old-session", 0, chunk).json()
        assert replay["replayed"] is True
        continued = _append(
            client, "old-session", FRAME64_LEN, frame64(1 << 63)
        ).json()
        assert continued["sequence_audit"]["last_sequence"] == 1 << 63


# ----------------------- leftovers of the old non-transactional migration ----


def test_stale_staging_table_next_to_old_sessions_is_rebuilt(db_path):
    """The previous release could leave a populated staging table behind.

    Its own run was killed after the copy but before the original table was
    dropped. The new transactional migration simply discards the stale
    staging table and rebuilds from the intact original.
    """
    create_audit_era_database(db_path)
    insert_audit_era_session(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(dbmod._sessions_table_sql(dbmod._MIGRATION_STAGING_TABLE))
        conn.execute(
            f"INSERT INTO {dbmod._MIGRATION_STAGING_TABLE} "
            "(session_id, protocol_id, version, received_offset, tail_b64, "
            " last_response) VALUES "
            "('stale-row', 'sensor-line64', 1, 0, '', '{}')"
        )
        conn.commit()
    finally:
        conn.close()

    dbmod.init_db(db_path)

    types = _wide_column_types(db_path)
    assert types["audit_last_sequence"] == "TEXT"
    assert types["audit_missing_total"] == "TEXT"
    assert dbmod._MIGRATION_STAGING_TABLE not in _table_names(db_path)
    probe = sqlite3.connect(db_path)
    try:
        ids = {r[0] for r in probe.execute("SELECT session_id FROM sessions")}
        assert ids == {"old-session"}  # stale staging copy discarded
        row = probe.execute(
            "SELECT audit_last_sequence, audit_missing_total FROM sessions"
        ).fetchone()
        assert row == (str(INT64_MAX), "5")
    finally:
        probe.close()


def test_staging_table_is_adopted_when_sessions_table_was_already_dropped(db_path):
    """Old release killed between DROP TABLE sessions and the RENAME.

    The fully populated staging table *is* the migrated result; adopting it
    rescues the historical sessions instead of creating an empty table.
    """
    staging = dbmod._MIGRATION_STAGING_TABLE
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(dbmod._sessions_table_sql(staging))
        conn.execute(
            f"INSERT INTO {staging} "
            "(session_id, protocol_id, version, received_offset, tail_b64, "
            " last_response, audit_enabled, audit_bit_width, "
            " audit_last_sequence, audit_missing_total) "
            "VALUES ('orphaned', 'sensor-line64', 1, 42, 'x', '{}', 1, 64, ?, ?)",
            (str(INT64_MAX), "5"),
        )
        conn.commit()
    finally:
        conn.close()
    assert "sessions" not in _table_names(db_path)

    dbmod.init_db(db_path)

    assert staging not in _table_names(db_path)
    probe = sqlite3.connect(db_path)
    try:
        types = {
            row[1]: row[2].upper()
            for row in probe.execute("PRAGMA table_info(sessions)")
        }
        assert types["audit_last_sequence"] == "TEXT"
        row = probe.execute(
            "SELECT session_id, received_offset, tail_b64, "
            "       audit_last_sequence, audit_missing_total "
            "FROM sessions"
        ).fetchone()
        assert row == ("orphaned", 42, "x", str(INT64_MAX), "5")
    finally:
        probe.close()
