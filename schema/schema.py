"""
SQLite-backed schema management.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


class SchemaManager:
    """Loads schemas from SQLite instead of filesystem defaults."""

    def __init__(self, db_url: Optional[str] = None):
        self.db = DatabaseManager(db_url=db_url)

    def get_schema(
        self,
        form_name: Optional[str] = None,
        schema_id: Optional[str] = None,
    ) -> Optional[Dict]:
        row = self.db.get_schema(form_name=form_name, schema_id=schema_id)
        return row["schema"] if row else None

    def list_schemas(self) -> List[Dict]:
        return self.db.list_registered_schemas()

    def add_schema(self, schema: Dict, schema_id: Optional[str] = None) -> str:
        return self.db.upsert_schema(schema, schema_id=schema_id, source="schema_manager")


def load_schema(
    form_name: Optional[str] = None,
    schema_id: Optional[str] = None,
    required: bool = True,
) -> Dict:
    manager = SchemaManager()
    schema = manager.get_schema(form_name=form_name, schema_id=schema_id)
    if schema is not None:
        return schema

    if required:
        target = f"form_name={form_name!r}" if form_name else f"schema_id={schema_id!r}"
        raise LookupError(f"Schema not found in SQLite store for {target}")
    return {}
