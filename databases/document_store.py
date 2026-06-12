"""
MongoDB document store.

Implements the six document databases of the data layer as collections of a
single MongoDB database:

  - original_texts: case_id, original_text
  - paragraphs:     case_id, paragraph_id, text
  - sentences:      case_id, sentence_id, text
  - propositions:   case_id, proposition_id, text
  - quotes:         case_id, quote_id, text
  - metadata:       case_id, citations, judge_name, case_year, court_type,
                    main_categories, page_rank_score

All write operations are idempotent (upserts / replace-on-rebuild) so the
build pipeline can be safely re-run on the same corpus.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from pymongo import ASCENDING, MongoClient, ReplaceOne
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from databases.chunking import Chunk
from databases.config import MongoConfig

logger = logging.getLogger(__name__)

# Maps the logical chunk level to (collection name, per-chunk id field).
_CHUNK_LEVELS: Dict[str, Dict[str, str]] = {
    "paragraphs": {"collection": "paragraphs", "id_field": "paragraph_id"},
    "sentences": {"collection": "sentences", "id_field": "sentence_id"},
    "propositions": {"collection": "propositions", "id_field": "proposition_id"},
    "quotes": {"collection": "quotes", "id_field": "quote_id"},
}

_METADATA_FIELDS = (
    "citations",
    "judge_name",
    "case_year",
    "court_type",
    "main_categories",
    "page_rank_score",
)


class MongoDocumentStore:
    """Unified read/write access to all MongoDB-backed databases."""

    def __init__(self, config: MongoConfig):
        self._config = config
        self._client = MongoClient(
            config.uri,
            serverSelectionTimeoutMS=config.server_selection_timeout_ms,
        )
        self._db = self._client[config.database_name]
        self._ensure_indexes()

    def ping(self) -> bool:
        """Return True when the MongoDB server is reachable."""
        try:
            self._client.admin.command("ping")
            return True
        except PyMongoError as exc:
            logger.error("MongoDB ping failed: %s", exc)
            return False

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def _ensure_indexes(self) -> None:
        """Create the unique indexes that enforce the logical schemas."""
        self._db["original_texts"].create_index(
            [("case_id", ASCENDING)], unique=True
        )
        self._db["metadata"].create_index([("case_id", ASCENDING)], unique=True)
        for level, spec in _CHUNK_LEVELS.items():
            self._db[spec["collection"]].create_index(
                [("case_id", ASCENDING), (spec["id_field"], ASCENDING)],
                unique=True,
            )

    def _chunk_collection(self, level: str) -> Collection:
        if level not in _CHUNK_LEVELS:
            raise ValueError(
                f"Unknown chunk level '{level}'. "
                f"Expected one of: {sorted(_CHUNK_LEVELS)}"
            )
        return self._db[_CHUNK_LEVELS[level]["collection"]]

    # ------------------------------------------------------------------
    # Original texts
    # ------------------------------------------------------------------

    def upsert_original_text(self, case_id: str, original_text: str) -> None:
        self._db["original_texts"].replace_one(
            {"case_id": case_id},
            {"case_id": case_id, "original_text": original_text},
            upsert=True,
        )

    def get_original_text(self, case_id: str) -> Optional[str]:
        document = self._db["original_texts"].find_one({"case_id": case_id})
        return document["original_text"] if document else None

    def iter_case_ids(self) -> Iterator[str]:
        """Iterate over all case ids in the knowledge base."""
        for document in self._db["original_texts"].find({}, {"case_id": 1}):
            yield document["case_id"]

    def count_cases(self) -> int:
        return self._db["original_texts"].count_documents({})

    # ------------------------------------------------------------------
    # Chunk collections (paragraphs / sentences / propositions / quotes)
    # ------------------------------------------------------------------

    def replace_chunks(self, level: str, case_id: str, chunks: Sequence[Chunk]) -> None:
        """Replace all chunks of one case at the given level.

        Deleting first keeps re-builds consistent when the new chunking
        produces fewer chunks than a previous run.
        """
        collection = self._chunk_collection(level)
        id_field = _CHUNK_LEVELS[level]["id_field"]

        collection.delete_many({"case_id": case_id})
        if not chunks:
            return
        collection.bulk_write(
            [
                ReplaceOne(
                    {"case_id": chunk.case_id, id_field: chunk.chunk_id},
                    {
                        "case_id": chunk.case_id,
                        id_field: chunk.chunk_id,
                        "text": chunk.text,
                    },
                    upsert=True,
                )
                for chunk in chunks
            ],
            ordered=False,
        )

    def get_chunks(self, level: str, case_id: str) -> List[Chunk]:
        """Return the ordered chunks of one case at the given level."""
        collection = self._chunk_collection(level)
        id_field = _CHUNK_LEVELS[level]["id_field"]
        return [
            Chunk(case_id=doc["case_id"], chunk_id=doc[id_field], text=doc["text"])
            for doc in collection.find({"case_id": case_id}).sort(id_field, ASCENDING)
        ]

    def iter_chunks(self, level: str) -> Iterator[Chunk]:
        """Stream every chunk at the given level across all cases."""
        collection = self._chunk_collection(level)
        id_field = _CHUNK_LEVELS[level]["id_field"]
        for doc in collection.find({}).sort(
            [("case_id", ASCENDING), (id_field, ASCENDING)]
        ):
            yield Chunk(
                case_id=doc["case_id"], chunk_id=doc[id_field], text=doc["text"]
            )

    def count_chunks(self, level: str) -> int:
        return self._chunk_collection(level).count_documents({})

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def upsert_metadata(self, case_id: str, **fields: Any) -> None:
        """Insert or partially update the metadata record of a case.

        Only known schema fields are accepted; unknown keyword arguments
        raise immediately to catch typos at the call site.
        """
        unknown = set(fields) - set(_METADATA_FIELDS)
        if unknown:
            raise ValueError(
                f"Unknown metadata fields: {sorted(unknown)}. "
                f"Allowed fields: {list(_METADATA_FIELDS)}"
            )
        self._db["metadata"].update_one(
            {"case_id": case_id},
            {"$set": {"case_id": case_id, **fields}},
            upsert=True,
        )

    def get_metadata(self, case_id: str) -> Optional[Dict[str, Any]]:
        document = self._db["metadata"].find_one(
            {"case_id": case_id}, {"_id": 0}
        )
        return document

    def get_metadata_bulk(self, case_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch metadata records for many cases in one round-trip."""
        cursor = self._db["metadata"].find(
            {"case_id": {"$in": list(case_ids)}}, {"_id": 0}
        )
        return {doc["case_id"]: doc for doc in cursor}

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def drop_all(self) -> None:
        """Drop every collection of the data layer. Destructive."""
        for name in ["original_texts", "metadata"] + [
            spec["collection"] for spec in _CHUNK_LEVELS.values()
        ]:
            self._db[name].drop()
        self._ensure_indexes()
        logger.warning("All MongoDB collections dropped and re-initialized.")
