from pathlib import Path

from pyprojroot import here

root_dir = str(here())
qdrant_binaries_dir = str(Path(root_dir) / "QdrantBinaries")
tmp_dir = "/tmp"