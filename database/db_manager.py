"""
Database Manager
================
SQLite-backed store for both runtime extraction records and schema metadata.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_TYPE_MAP: Dict[str, str] = {
    "string": "TEXT",
    "date": "TEXT",
    "number": "REAL",
    "boolean": "INTEGER",
    "currency": "TEXT",
    "email": "TEXT",
}

_SYSTEM_COLS = [
    ("_record_id", "TEXT PRIMARY KEY"),
    ("_schema_version", "TEXT"),
    ("_ingested_at", "TEXT"),
    ("_source_doc", "TEXT"),
    ("_confidence_avg", "REAL"),
]


def _table_name(form_name: str) -> str:
    return re.sub(r"[^\w]", "_", form_name).lower().strip("_")


def _col_def(field_name: str, field_meta: Dict[str, Any]) -> str:
    affinity = _TYPE_MAP.get(field_meta.get("type", "string"), "TEXT")
    return f"{field_name} {affinity}"


class DatabaseManager:
    """Schema-driven SQLite database manager."""

    def __init__(self, db_url: Optional[str] = None):
        if db_url is None:
            from config.config import get_config

            db_url = str(get_config().paths.data_dir / "etl.db")

        if db_url.startswith("sqlite:///"):
            db_url = db_url[len("sqlite:///"):]

        self.db_path = Path(db_url)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("[DB] Using database: %s", self.db_path)
        self._created_tables: set[str] = set()
        self._init_schema_table()
        self._init_meta_table()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema_table(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS _schemas (
                    schema_id   TEXT PRIMARY KEY,
                    form_name   TEXT NOT NULL UNIQUE,
                    version     TEXT,
                    description TEXT,
                    schema_json TEXT NOT NULL,
                    source      TEXT,
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _init_meta_table(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS _schema_registry_meta (
                    form_name      TEXT PRIMARY KEY,
                    table_name     TEXT NOT NULL,
                    schema_json    TEXT NOT NULL,
                    schema_id      TEXT,
                    registered_at  TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def upsert_schema(
        self,
        schema: Dict[str, Any],
        schema_id: Optional[str] = None,
        source: str = "runtime",
    ) -> str:
        form_name = (schema or {}).get("form_name")
        if not form_name:
            raise ValueError("Schema missing required 'form_name'")

        now = datetime.utcnow().isoformat()
        existing = self.get_schema(form_name=form_name)
        resolved_schema_id = schema_id or (existing or {}).get("schema_id") or str(uuid.uuid4())

        created_at = (existing or {}).get("created_at", now)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO _schemas
                    (schema_id, form_name, version, description, schema_json, source, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(form_name) DO UPDATE SET
                    version     = excluded.version,
                    description = excluded.description,
                    schema_json = excluded.schema_json,
                    source      = excluded.source,
                    updated_at  = excluded.updated_at
                """,
                (
                    resolved_schema_id,
                    form_name,
                    schema.get("version", "1.0"),
                    schema.get("description", ""),
                    json.dumps(schema),
                    source,
                    created_at,
                    now,
                ),
            )
            conn.commit()

        logger.info("[DB] Schema '%s' stored as %s", form_name, resolved_schema_id)
        return resolved_schema_id

    def get_schema(
        self,
        form_name: Optional[str] = None,
        schema_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not form_name and not schema_id:
            raise ValueError("Either form_name or schema_id must be provided")

        clauses: List[str] = []
        params: List[str] = []
        if form_name:
            clauses.append("form_name = ?")
            params.append(form_name)
        if schema_id:
            clauses.append("schema_id = ?")
            params.append(schema_id)

        query = (
            "SELECT schema_id, form_name, version, description, schema_json, source, created_at, updated_at "
            "FROM _schemas WHERE " + " AND ".join(clauses) + " LIMIT 1"
        )
        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()
            if row:
                schema = json.loads(row["schema_json"])
                schema.setdefault("schema_id", row["schema_id"])
                return {
                    "schema_id": row["schema_id"],
                    "form_name": row["form_name"],
                    "version": row["version"],
                    "description": row["description"],
                    "schema_json": row["schema_json"],
                    "source": row["source"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "schema": schema,
                }

        return self._get_legacy_schema(form_name=form_name, schema_id=schema_id)

    def _get_legacy_schema(
        self,
        form_name: Optional[str] = None,
        schema_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[str] = []
        if form_name:
            clauses.append("form_name = ?")
            params.append(form_name)
        if schema_id:
            clauses.append("schema_id = ?")
            params.append(schema_id)
        if not clauses:
            return None

        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT form_name, schema_json, schema_id, registered_at "
                    "FROM _schema_registry_meta WHERE "
                    + " AND ".join(clauses)
                    + " LIMIT 1",
                    params,
                ).fetchone()
        except sqlite3.OperationalError:
            return None

        if not row:
            return None

        schema = json.loads(row["schema_json"])
        resolved_schema_id = row["schema_id"] or str(uuid.uuid4())
        schema.setdefault("schema_id", resolved_schema_id)
        return {
            "schema_id": resolved_schema_id,
            "form_name": row["form_name"],
            "version": schema.get("version", "1.0"),
            "description": schema.get("description", ""),
            "schema_json": row["schema_json"],
            "source": "legacy_meta",
            "created_at": row["registered_at"],
            "updated_at": row["registered_at"],
            "schema": schema,
        }

    def list_registered_schemas(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT schema_id, form_name, version, description, source, created_at, updated_at
                FROM _schemas
                ORDER BY form_name
                """
            ).fetchall()
        if rows:
            return [dict(row) for row in rows]

        try:
            with self._connect() as conn:
                legacy_rows = conn.execute(
                    """
                    SELECT schema_id, form_name, '' AS version, '' AS description,
                           'legacy_meta' AS source, registered_at AS created_at,
                           registered_at AS updated_at
                    FROM _schema_registry_meta
                    ORDER BY form_name
                    """
                ).fetchall()
                return [dict(row) for row in legacy_rows]
        except sqlite3.OperationalError:
            return []

    def ensure_table(self, schema: Dict[str, Any], schema_id: Optional[str] = None) -> str:
        form_name = schema.get("form_name", "document")
        table_name = _table_name(form_name)

        col_defs = [f"{name} {dtype}" for name, dtype in _SYSTEM_COLS]
        for field_name, meta in schema.get("fields", {}).items():
            col_defs.append(_col_def(field_name, meta))

        create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(col_defs)})"

        with self._connect() as conn:
            conn.execute(create_sql)
            existing_cols = {
                row[1].lower() for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            }
            for field_name, meta in schema.get("fields", {}).items():
                if field_name.lower() not in existing_cols:
                    affinity = _TYPE_MAP.get(meta.get("type", "string"), "TEXT")
                    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {field_name} {affinity}")
                    logger.info("[DB] Added column '%s' to '%s'", field_name, table_name)

            conn.execute(
                """
                INSERT INTO _schema_registry_meta
                    (form_name, table_name, schema_json, schema_id, registered_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(form_name) DO UPDATE SET
                    schema_json   = excluded.schema_json,
                    schema_id     = excluded.schema_id,
                    registered_at = excluded.registered_at
                """,
                (
                    form_name,
                    table_name,
                    json.dumps(schema),
                    schema_id or "",
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()

        try:
            self.upsert_schema(schema, schema_id=schema_id, source="db_population")
        except Exception as exc:
            logger.warning("[DB] Failed to mirror schema into _schemas: %s", exc)

        self._created_tables.add(table_name)
        logger.info("[DB] Table '%s' ready", table_name)
        return table_name

    def insert_record(
        self,
        schema: Dict[str, Any],
        fields: Dict[str, Any],
        schema_id: Optional[str] = None,
        source_doc: Optional[str] = None,
        confidence_avg: float = 0.0,
    ) -> str:
        table_name = self.ensure_table(schema, schema_id)
        record_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        row: Dict[str, Any] = {
            "_record_id": record_id,
            "_schema_version": schema.get("version", "1.0"),
            "_ingested_at": now,
            "_source_doc": source_doc or "",
            "_confidence_avg": round(confidence_avg, 4),
        }
        for field_name, value in fields.items():
            if isinstance(value, bool):
                row[field_name] = int(value)
            elif value == "NaN" or value is None:
                row[field_name] = None
            else:
                row[field_name] = value

        columns = ", ".join(row.keys())
        placeholders = ", ".join("?" for _ in row)
        sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

        with self._connect() as conn:
            conn.execute(sql, list(row.values()))
            conn.commit()

        logger.info("[DB] Inserted record %s into '%s'", record_id, table_name)
        return record_id

    def query_records(
        self,
        form_name: str,
        limit: int = 100,
        where: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        table_name = _table_name(form_name)
        sql = f"SELECT * FROM {table_name}"
        if where:
            sql += f" WHERE {where}"
        sql += f" ORDER BY _ingested_at DESC LIMIT {limit}"
        try:
            with self._connect() as conn:
                rows = conn.execute(sql).fetchall()
                return [dict(row) for row in rows]
        except sqlite3.OperationalError as exc:
            logger.warning("[DB] query_records failed for '%s': %s", table_name, exc)
            return []

    def list_tables(self) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE '\\_%' ESCAPE '\\'"
            ).fetchall()
            return [row[0] for row in rows]

    def table_exists(self, form_name: str) -> bool:
        table_name = _table_name(form_name)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            ).fetchone()
            return row is not None

    def record_count(self, form_name: str) -> int:
        table_name = _table_name(form_name)
        try:
            with self._connect() as conn:
                return conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        except sqlite3.OperationalError:
            return 0
