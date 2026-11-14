"""Load review CSVs, keep a persistent Chroma index in sync, and expose review stats."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

import config


@dataclass
class RestaurantStats:
    name: str
    count: int
    average_rating: float | None
    rating_counts: dict[int, int]
    first_date: str
    last_date: str

    def summary(self) -> str:
        avg = "n/a" if self.average_rating is None else f"{self.average_rating:.2f}"
        dist = ", ".join(f"{s} stars: {self.rating_counts.get(s, 0)}" for s in range(5, 0, -1))
        return (
            f"{self.name}: {self.count} reviews, average rating {avg}/5 "
            f"({dist}), dated {self.first_date} to {self.last_date}"
        )


# ----------------------------------------------------------------------------
# CSV loading
# ----------------------------------------------------------------------------
def _clean(value) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _pretty_name(path: Path) -> str:
    return re.sub(r"[_\-]+", " ", path.stem).strip().title()


def list_csv_files(data_dir: Path = config.DATA_DIR) -> list[Path]:
    files = sorted(data_dir.glob("*.csv")) if data_dir.is_dir() else []
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. Add one or more review CSVs "
            f"with columns: {', '.join(sorted(config.REQUIRED_COLUMNS))} "
            "(and optionally Restaurant)."
        )
    return files


def read_reviews(csv_path: Path) -> pd.DataFrame:
    """Read one CSV, validate it, and normalise the columns."""
    frame = pd.read_csv(csv_path)
    frame.columns = [str(c).strip() for c in frame.columns]
    missing = config.REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns: {', '.join(sorted(missing))}")

    if "Restaurant" not in frame.columns:
        frame["Restaurant"] = _pretty_name(csv_path)
    frame["Restaurant"] = frame["Restaurant"].map(_clean).replace("", _pretty_name(csv_path))
    frame["Rating"] = pd.to_numeric(frame["Rating"], errors="coerce")
    frame["Date"] = frame["Date"].map(_clean)
    frame["Title"] = frame["Title"].map(_clean)
    frame["Review"] = frame["Review"].map(_clean)
    return frame[(frame["Title"] != "") | (frame["Review"] != "")].reset_index(drop=True)


def _split_text(text: str) -> list[str]:
    """Return the text as-is when short; otherwise overlapping sentence-aligned chunks."""
    limit, overlap = config.MAX_CHUNK_CHARS, config.CHUNK_OVERLAP
    if len(text) <= limit:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence[:limit]  # guard against a single huge "sentence"
        if current and len(current) + len(sentence) + 1 > limit:
            chunks.append(current)
            current = current[-overlap:].split(" ", 1)[-1]  # start on a word boundary
        current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def _frame_to_documents(frame: pd.DataFrame, source: str) -> list[Document]:
    documents: list[Document] = []
    for row_number, row in frame.iterrows():
        text = f"{row['Title']}. {row['Review']}".strip(". ")
        rating = None if pd.isna(row["Rating"]) else float(row["Rating"])
        for part, chunk in enumerate(_split_text(text)):
            documents.append(
                Document(
                    # The restaurant name is embedded so queries naming it match well.
                    page_content=f"[{row['Restaurant']}] {chunk}",
                    metadata={
                        "restaurant": row["Restaurant"],
                        "title": row["Title"],
                        "rating": rating if rating is not None else -1.0,
                        "date": row["Date"] or "unknown",
                        "source": source,
                        "row": int(row_number),  # type: ignore[arg-type]
                    },
                    id=f"{source}:{row_number}:{part}",
                )
            )
    return documents


def load_all_frames() -> pd.DataFrame:
    frames = [read_reviews(p).assign(_source=p.name) for p in list_csv_files()]
    return pd.concat(frames, ignore_index=True)


def compute_stats(frame: pd.DataFrame | None = None) -> dict[str, RestaurantStats]:
    frame = load_all_frames() if frame is None else frame
    stats: dict[str, RestaurantStats] = {}
    for name, group in frame.groupby("Restaurant"):
        ratings = group["Rating"].dropna()
        dates = sorted(d for d in group["Date"] if d)
        stats[str(name)] = RestaurantStats(
            name=str(name),
            count=len(group),
            average_rating=float(ratings.mean()) if len(ratings) else None,
            rating_counts={int(k): int(v) for k, v in ratings.round().value_counts().items()},
            first_date=dates[0] if dates else "unknown",
            last_date=dates[-1] if dates else "unknown",
        )
    return dict(sorted(stats.items()))


# ----------------------------------------------------------------------------
# Index management
# ----------------------------------------------------------------------------
def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest() -> dict:
    try:
        return json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_manifest(manifest: dict) -> None:
    config.DB_DIR.mkdir(parents=True, exist_ok=True)
    config.MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _add_in_batches(store: Chroma, documents: list[Document]) -> None:
    size = config.EMBED_BATCH_SIZE
    for start in range(0, len(documents), size):
        batch = documents[start : start + size]
        store.add_documents(batch, ids=[d.id for d in batch])  # type: ignore[misc]
        print(f"  embedded {min(start + size, len(documents))}/{len(documents)}", end="\r")
    print()


def build_vector_store(force_rebuild: bool = False) -> Chroma:
    """Open the Chroma store and incrementally sync it with the CSV files in data/."""
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL, base_url=config.OLLAMA_BASE_URL)
    files = list_csv_files()
    hashes = {p.name: _file_sha256(p) for p in files}

    manifest = _read_manifest()
    settings = {"embedding_model": config.EMBEDDING_MODEL, "index_version": config.INDEX_VERSION}
    if force_rebuild or manifest.get("settings") != settings:
        manifest = {}
        _reset_collection(embeddings)

    store = Chroma(
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.DB_DIR),
        embedding_function=embeddings,
    )

    indexed: dict[str, str] = manifest.get("files", {})

    # Remove data for CSVs that were deleted or changed.
    for name in list(indexed):
        if name not in hashes or indexed[name] != hashes[name]:
            existing = store.get(where={"source": name}).get("ids", [])
            if existing:
                store.delete(ids=existing)
            indexed.pop(name)
            print(f"Removed stale index entries for {name}.")

    # Add new or changed CSVs.
    for path in files:
        if path.name in indexed:
            continue
        documents = _frame_to_documents(read_reviews(path), path.name)
        if not documents:
            print(f"Skipping {path.name}: no usable rows.")
            continue
        print(f"Indexing {path.name} ({len(documents)} chunks)...")
        _add_in_batches(store, documents)
        indexed[path.name] = hashes[path.name]
        _write_manifest({"settings": settings, "files": indexed})

    _write_manifest({"settings": settings, "files": indexed})
    return store


def _reset_collection(embeddings: OllamaEmbeddings) -> None:
    try:
        Chroma(
            collection_name=config.COLLECTION_NAME,
            persist_directory=str(config.DB_DIR),
            embedding_function=embeddings,
        ).delete_collection()
    except Exception:
        pass  # Nothing to reset on a first run.
