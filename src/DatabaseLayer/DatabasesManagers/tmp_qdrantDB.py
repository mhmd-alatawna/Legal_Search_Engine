import atexit
import json
import logging
import os
import pickle
import platform
import shutil
import subprocess
import threading
import time
import uuid
import warnings
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from tqdm import tqdm
from src.DatabaseLayer.Processors.Schema import Schema


logger = logging.getLogger(__name__)


class QdrantDB:
    """One managed Qdrant database containing independently registered collections."""

    metadata_directory = None
    metadata_directory_file = "metadata_directory.json"
    _shared_process = None
    _shared_client = None
    _shared_db_path = None
    _init_lock = threading.Lock()

    def __init__(self, db_path, binaries_dir):
        requested_path = Path(db_path).resolve()

        # This project intentionally uses one Qdrant database per Python process.
        with QdrantDB._init_lock:
            if QdrantDB._shared_db_path is None:
                QdrantDB._shared_db_path = requested_path
            elif QdrantDB._shared_db_path != requested_path:
                warnings.warn(f"QdrantDB is already using '{QdrantDB._shared_db_path}'. The new path '{requested_path}' is ignored.", RuntimeWarning, stacklevel=2)

        self.db_path = QdrantDB._shared_db_path
        self.snapshots_dir = self.db_path / "snapshots"
        Path(self.db_path).mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        self._ensure_server_running(binaries_dir)
        self.client = QdrantDB._shared_client
        self.__load_metadata_directory()
        self._recover_registered_collections()

    def _ensure_server_running(self, binaries_dir):
        with QdrantDB._init_lock:
            if QdrantDB._shared_client is not None:
                return

            binary_name = "qdrant.exe" if platform.system() == "Windows" else "qdrant"
            binary_path = Path(binaries_dir).resolve() / binary_name
            if not binary_path.exists():
                raise FileNotFoundError(f"Qdrant engine binary missing at: {binary_path}")

            server_env = os.environ.copy()
            server_env["QDRANT__STORAGE__STORAGE_PATH"] = str(self.db_path)
            server_env["QDRANT__SERVICE__HOST"] = "127.0.0.1"
            server_env["QDRANT__SERVICE__HTTP_PORT"] = "6333"
            server_env["QDRANT__SERVICE__GRPC_PORT"] = "6334"
            server_env["QDRANT__TELEMETRY_DISABLED"] = "true"
            server_env["QDRANT__STORAGE__SNAPSHOTS_PATH"] = str(self.snapshots_dir)

            QdrantDB._shared_process = subprocess.Popen(
                [str(binary_path)],
                env=server_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            QdrantDB._shared_client = QdrantClient(
                host="127.0.0.1",
                port=6333,
                grpc_port=6334,
                prefer_grpc=True,
                timeout=300,
            )

            last_error = None
            for _ in range(30):
                try:
                    QdrantDB._shared_client.get_collections()
                    break
                except Exception as error:
                    last_error = error
                    time.sleep(0.5)
            else:
                QdrantDB._cleanup_server()
                raise RuntimeError("Qdrant server did not become ready") from last_error

            atexit.register(QdrantDB._cleanup_server)

    def _recover_registered_collections(self):
        for collection_name in QdrantDB.metadata_directory.keys():
            if self.client.collection_exists(collection_name):
                continue
            else:
                snapshot_path = (self.snapshots_dir / f"{collection_name}.snapshot").resolve()
                if not snapshot_path.exists():
                    logger.warning(f"Collection {collection_name} is registered, but neither live storage nor a snapshot exists.")
                    continue
                logger.info(f"Recovering Qdrant collection {collection_name} from {snapshot_path}")
                self.client.recover_snapshot(collection_name=collection_name, location=f"file:///{snapshot_path.as_posix()}")
        return None

    def _create_collection(self, collection_name, embedder):
        """Create or validate a collection and remember how its records are encoded."""
        registered = QdrantDB.metadata_directory.get(collection_name,None)

        if self.client.collection_exists(collection_name):
            if registered is None:
                raise RuntimeError("ERROR : mismatch between server and metadata")
            elif registered != embedder.schema:
                logger.warning(f"Collection '{collection_name}' is already registered with different schema , deleting it.")
                #     TODO : add collection deletion logic
            else :
                print(f"collection {collection_name} already exists , working with it")
                return
        else :
            print(f"{collection_name} is new , Creating new collection")


        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=embedder.schema.vector_size,
                distance=qmodels.Distance.COSINE,
                on_disk=True,
            ),
            optimizers_config=qmodels.OptimizersConfigDiff(indexing_threshold=0),
        )
        self.__update_metadata_directory(collection_name, embedder.schema)

    @staticmethod
    def _record_dict(record, fields):
        if len(record) != len(fields):
            raise ValueError(f"record and fields sizes are different : {len(record)} != {len(fields)}")
        return {fields[i][0] : record[i] for i in range(len(fields))}

    def index_documents(self, documents, collection_name, embedder, batch_size = 1024):
        self._create_collection(collection_name, embedder)

        fields = embedder.schema.fields
        embedding_field = embedder.schema.main_field
        primary_keys = embedder.schema.primary_keys

        progress = tqdm(range(0, len(documents), batch_size), desc=f"Vectorizing -> {collection_name}")
        for start in progress:
            batch = documents[start:start + batch_size]
            batch = [self._record_dict(record, fields) for record in batch]
            vectors = embedder.encode([str(record[embedding_field]) for record in batch])
            if len(vectors) != len(batch):
                raise RuntimeError(f"Embedding processor returned an unexpected number of vectors")

            points = []
            for record, vector in zip(batch, vectors):
                only_primary_keys_records = {field : value for field,value in record.items() if field in primary_keys}
                stable_record = json.dumps(
                    only_primary_keys_records,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                    separators=(",", ":"),
                )
                point_id = str(
                    uuid.uuid5(uuid.NAMESPACE_URL, f"{collection_name}/{stable_record}")
                )
                payload = {
                    field: value
                    for field, value in record.items()
                    if field != embedding_field
                }
                points.append(qmodels.PointStruct(id=point_id, vector=vector, payload=payload))

            self.client.upsert(collection_name=collection_name, points=points)

        self.client.update_collection(
            collection_name,
            optimizer_config=qmodels.OptimizersConfigDiff(indexing_threshold=20000),
        )
        self._create_snapshot(collection_name)

    def _create_snapshot(self, collection_name):
        snapshot = self.client.create_snapshot(collection_name)

        source_dir = self.snapshots_dir / collection_name
        source_path = source_dir / snapshot.name
        target_path = self.snapshots_dir / f"{collection_name}.snapshot"

        if source_path.exists():
            shutil.move(str(source_path), str(target_path))
            if source_dir.exists() and not any(source_dir.iterdir()):
                source_dir.rmdir()
            logger.info(f"Successfully created snapshot at {target_path}")
        else:
            # 5. Failsafe alert
            logger.error(
                "Qdrant reported snapshot '%s' created, but it was not found at %s. "
                "Ensure the subprocess has write permissions to this path.",
                snapshot.name, source_path)

    @staticmethod
    def _format_hit(point, schema):
        payload = point.payload or {}
        result_fields = [
            field
            for field in schema.fields
            if field != schema.main_field
        ]
        return tuple(payload[field[0]] for field in result_fields if field[0] != schema.main_field) + (float(point.score),)

    def search(self, query, top_k, collection_name, embedder):
        if embedder.schema !=  QdrantDB.metadata_directory.get(collection_name, None):
            raise ValueError(f"Embedders have different collection schemas")

        query_vector = embedder.encode([str(query)])[0]
        response = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        return [self._format_hit(point, QdrantDB.metadata_directory[collection_name]) for point in response.points]

    def search_batch(
        self,
        queries, top_k, collection_name, embedder):
        if not queries:
            return []

        if embedder.schema != QdrantDB.metadata_directory.get(collection_name, None):
            raise ValueError(f"Embedders have different collection schemas")

        requests = [
            qmodels.QueryRequest(query=vector, limit=top_k, with_payload=True)
            for vector in embedder.encode([str(q) for q in queries])
        ]
        responses = self.client.query_batch_points(
            collection_name=collection_name,
            requests=requests,
        )
        return [
            [self._format_hit(point, QdrantDB.metadata_directory[collection_name]) for point in response.points]
            for response in responses
        ]

    @classmethod
    def _cleanup_server(cls) -> None:
        if cls._shared_client is not None:
            try:
                cls._shared_client.close()
            except Exception:
                logger.exception("Failed to close the Qdrant client")
            cls._shared_client = None

        if cls._shared_process is not None:
            cls._shared_process.terminate()
            try:
                cls._shared_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls._shared_process.kill()
                cls._shared_process.wait()
            cls._shared_process = None

        # The only database instance is closed, so a later instance may use a
        # database directory that has been moved to a new path.
        cls._shared_db_path = None
        cls.metadata_directory = None

    def close(self) -> None:
        self._cleanup_server()

    def __load_metadata_directory(self):
        file_name = f"{self.db_path}/{QdrantDB.metadata_directory_file}"
        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                QdrantDB.metadata_directory = pickle.load(f)
        else :
            QdrantDB.metadata_directory = {}

    def __update_metadata_directory(self, collection_name, collection_definition):
        file_name = f"{self.db_path}/{QdrantDB.metadata_directory_file}"
        QdrantDB.metadata_directory[collection_name] = collection_definition
        with open(file_name, "wb") as f:
            pickle.dump(QdrantDB.metadata_directory, f)

    def __enter__(self) -> "QdrantDB":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @staticmethod
    def get_schema(collection_name):
        return QdrantDB.metadata_directory.get(collection_name, None)
