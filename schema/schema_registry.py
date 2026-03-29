"""
Schema Registry
===============
Stores known document schemas with dense embeddings of their field
descriptions.  Provides semantic similarity search so the schema-resolution
agent can find the closest matching schema for a previously-unseen document.

Storage layout  (under data/schema_registry/)
----------------------------------------------
  index.json          list of {schema_id, form_name, field_names, registered_at}
  <schema_id>.json    full schema dict
  <schema_id>.npy     stacked float32 embedding array (one row per field)

Embedding model: sentence-transformers/all-MiniLM-L6-v2  (~80 MB)
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_SIM_THRESHOLD = 0.70          # cosine sim floor for "compatible" match


# ---------------------------------------------------------------------------
# Embedding helper (lazy-loaded singleton)
# ---------------------------------------------------------------------------

_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"[Registry] Loading embedding model {_EMBED_MODEL} …")
            _embedder = SentenceTransformer(_EMBED_MODEL)
            logger.info("[Registry] Embedder ready")
        except Exception as exc:
            logger.error(f"[Registry] Failed to load embedder: {exc}")
            raise
    return _embedder


def _embed_schema(schema: Dict) -> np.ndarray:
    """
    Embed a schema as a matrix of shape (num_fields, embed_dim).
    Each row = embedding of "field_name: description".
    The schema embedding used for similarity is the centroid (mean row).
    """
    sentences = []
    for fname, meta in schema.get("fields", {}).items():
        desc = meta.get("description", fname)
        sentences.append(f"{fname}: {desc}")
    if not sentences:
        sentences = [schema.get("form_name", "unknown schema")]
    emb = _get_embedder().encode(sentences, normalize_embeddings=True)
    return np.array(emb, dtype=np.float32)


def _centroid(emb: np.ndarray) -> np.ndarray:
    """Unit-normalised centroid of a set of embeddings."""
    c = emb.mean(axis=0)
    norm = np.linalg.norm(c)
    return c / norm if norm > 0 else c


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two unit-norm vectors."""
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Registry class
# ---------------------------------------------------------------------------

class SchemaRegistry:
    """
    Persistent registry of document schemas with embedding-based retrieval.

    Parameters
    ----------
    registry_dir : Path
        Directory where index.json, schema JSONs and .npy files live.
    sim_threshold : float
        Minimum cosine similarity to consider a schema "compatible".
    """

    def __init__(
        self,
        registry_dir: Optional[Path] = None,
        sim_threshold: float = _SIM_THRESHOLD,
    ):
        if registry_dir is None:
            from config.config import get_config
            registry_dir = get_config().paths.data_dir / "schema_registry"
        self.registry_dir  = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.sim_threshold = sim_threshold

        self._index: List[Dict] = []
        self._load_index()

    # ------------------------------------------------------------------
    # Index persistence
    # ------------------------------------------------------------------

    def _index_path(self) -> Path:
        return self.registry_dir / "index.json"

    def _load_index(self) -> None:
        p = self._index_path()
        if p.exists():
            try:
                self._index = json.loads(p.read_text())
                logger.info(f"[Registry] Loaded {len(self._index)} schemas from index")
            except Exception as exc:
                logger.warning(f"[Registry] Could not read index: {exc}")
                self._index = []

    def _save_index(self) -> None:
        self._index_path().write_text(json.dumps(self._index, indent=2))

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, schema: Dict, overwrite: bool = False) -> str:
        """
        Register a schema and persist its embedding.

        Returns the schema_id (UUID string).
        """
        form_name = schema.get("form_name", "unknown")

        # Check for existing entry with same form_name
        existing = next((e for e in self._index if e["form_name"] == form_name), None)
        if existing and not overwrite:
            logger.info(f"[Registry] '{form_name}' already registered — skipping")
            return existing["schema_id"]

        schema_id = existing["schema_id"] if existing else str(uuid.uuid4())

        # Compute and save embedding
        emb = _embed_schema(schema)
        np.save(str(self.registry_dir / f"{schema_id}.npy"), emb)

        # Save full schema JSON
        (self.registry_dir / f"{schema_id}.json").write_text(
            json.dumps(schema, indent=2)
        )

        # Update index
        entry = {
            "schema_id":      schema_id,
            "form_name":      form_name,
            "field_names":    list(schema.get("fields", {}).keys()),
            "registered_at":  datetime.utcnow().isoformat(),
        }
        if existing:
            idx = next(i for i, e in enumerate(self._index) if e["schema_id"] == schema_id)
            self._index[idx] = entry
        else:
            self._index.append(entry)

        self._save_index()
        try:
            from database.db_manager import DatabaseManager

            DatabaseManager().upsert_schema(schema, schema_id=schema_id, source="schema_registry")
        except Exception as exc:
            logger.warning(f"[Registry] Failed to mirror schema to SQLite: {exc}")
        logger.info(f"[Registry] Registered '{form_name}' as {schema_id}")
        return schema_id

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def find_similar(
        self,
        query_schema_or_fields: Dict,
        top_k: int = 3,
    ) -> List[Tuple[float, str, Dict]]:
        """
        Return top-k schemas most similar to the query.

        Parameters
        ----------
        query_schema_or_fields :
            Either a full schema dict (with "fields" key) or a plain
            {field_name: value} dict extracted from a document.
        top_k : int

        Returns
        -------
        List of (similarity_score, schema_id, schema_dict), highest score first.
        Only entries above self.sim_threshold are returned.
        """
        if not self._index:
            return []

        # Build query embedding
        if "fields" in query_schema_or_fields:
            q_emb = _centroid(_embed_schema(query_schema_or_fields))
        else:
            # Plain field dict — embed field names as sentences
            emb = _get_embedder().encode(
                list(query_schema_or_fields.keys()), normalize_embeddings=True
            )
            q_emb = _centroid(np.array(emb, dtype=np.float32))

        scores: List[Tuple[float, str, Dict]] = []
        for entry in self._index:
            sid  = entry["schema_id"]
            npy  = self.registry_dir / f"{sid}.npy"
            sj   = self.registry_dir / f"{sid}.json"
            if not npy.exists() or not sj.exists():
                continue
            stored_emb    = np.load(str(npy))
            stored_centroid = _centroid(stored_emb)
            sim = _cosine(q_emb, stored_centroid)
            if sim >= self.sim_threshold:
                schema = json.loads(sj.read_text())
                scores.append((sim, sid, schema))

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]

    def get_schema(self, schema_id: str) -> Optional[Dict]:
        p = self.registry_dir / f"{schema_id}.json"
        if not p.exists():
            return None
        return json.loads(p.read_text())

    def list_schemas(self) -> List[Dict]:
        return list(self._index)

    def schema_count(self) -> int:
        return len(self._index)
