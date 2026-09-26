"""ChatPhone retrieval service: semantic search over Amazon smartphone reviews.

Local, standalone version of the search API from notebooks/01_preprocessing_and_search_api.ipynb
(which ran on Google Colab behind an ngrok tunnel). Same pipeline and same response format:

    items + reviews (joined on asin) -> full text -> Sentence-BERT embeddings -> FAISS index
    POST /search {"text": "...", "k": 20, "sentiment": null | "positive" | "negative"}
      -> {"results": [{"brand", "model", "review", ...}]}

The embeddings and indexes are computed once and cached in ``--cache-dir``.

Usage:
    python search_api/search_api.py --data-dir data --cache-dir cache
    python search_api/search_api.py --limit 3000      # quick test on a sample of reviews
"""
import argparse
import logging
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_NAME = "all-MiniLM-L6-v2"
ITEMS_FILE = "20190928-items.csv"
REVIEWS_FILE = "20190928-reviews.csv"
SENTIMENT_THRESHOLD = 0.05  # VADER compound score


def load_dataset(data_dir: Path) -> pd.DataFrame:
    """Same preprocessing as the notebook: join, keep useful columns, drop missing values."""
    items = pd.read_csv(data_dir / ITEMS_FILE)
    reviews = pd.read_csv(data_dir / REVIEWS_FILE)
    df = pd.merge(items, reviews, how="right", on="asin")
    df = df[["asin", "brand", "title_x", "rating_x", "totalReviews", "prices",
             "rating_y", "verified", "title_y", "body"]]
    df = df.dropna().reset_index(drop=True)
    df["full_text"] = (
        "Brand: " + df["brand"].astype(str) + ". "
        + "Price: " + df["prices"].astype(str) + ". "
        + "Product: " + df["title_x"].astype(str) + ". "
        + "Review Title: " + df["title_y"].astype(str) + ". "
        + "Review: " + df["body"].astype(str) + "."
    )
    return df


def build_or_load(data_dir: Path, cache_dir: Path, embedder: SentenceTransformer, limit: Optional[int] = None):
    """Return (df, embeddings), computing and caching them on first run."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit{limit}" if limit else ""
    df_path, emb_path = cache_dir / f"df_fulltext{suffix}.pkl", cache_dir / f"embeddings{suffix}.npy"

    if df_path.exists() and emb_path.exists():
        logging.info("Loading cached data from %s", cache_dir)
        return pd.read_pickle(df_path), np.load(emb_path)

    logging.info("Building the dataset from %s", data_dir)
    df = load_dataset(data_dir)
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=0).reset_index(drop=True)
    logging.info("Encoding %d reviews with %s (first run only)...", len(df), MODEL_NAME)
    embeddings = embedder.encode(df["full_text"].tolist(), batch_size=256,
                                 show_progress_bar=True, convert_to_numpy=True).astype("float32")
    analyzer = SentimentIntensityAnalyzer()
    df["sentiment_score"] = df["body"].fillna("").map(lambda t: analyzer.polarity_scores(t)["compound"])
    df.to_pickle(df_path)
    np.save(emb_path, embeddings)
    return df, embeddings


def make_index(vectors: np.ndarray) -> faiss.IndexFlatL2:
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index


def create_app(df: pd.DataFrame, embeddings: np.ndarray, embedder: SentenceTransformer) -> FastAPI:
    pos_ids = np.where(df["sentiment_score"] > SENTIMENT_THRESHOLD)[0]
    neg_ids = np.where(df["sentiment_score"] < -SENTIMENT_THRESHOLD)[0]
    indexes = {
        None: (make_index(embeddings), np.arange(len(df))),
        "positive": (make_index(embeddings[pos_ids]), pos_ids),
        "negative": (make_index(embeddings[neg_ids]), neg_ids),
    }
    logging.info("FAISS indexes ready: all=%d, positive=%d, negative=%d",
                 len(df), len(pos_ids), len(neg_ids))

    app = FastAPI(title="ChatPhone search API")

    class Query(BaseModel):
        text: str
        k: int = 50
        sentiment: Optional[str] = None  # None, "positive" or "negative"

    @app.get("/health")
    def health():
        return {"status": "ok", "reviews": len(df)}

    @app.post("/search")
    def search(q: Query):
        index, ids = indexes.get(q.sentiment, indexes[None])
        q_emb = embedder.encode([q.text], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        distances, hits = index.search(q_emb, min(q.k, index.ntotal))
        results = []
        for dist, hit in zip(distances[0], hits[0]):
            row = df.iloc[ids[hit]]
            results.append({
                "brand": row["brand"],
                "model": row["title_x"],
                "review": row["body"],
                "price": row["prices"],
                "rating": float(row["rating_y"]),
                "sentiment": float(row["sentiment_score"]),
                "distance": float(dist),
            })
        return {"results": results}

    return app


def main():
    parser = argparse.ArgumentParser(description="ChatPhone semantic search API")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--limit", type=int, default=None,
                        help="Index only a random sample of N reviews (quick test; the full dataset takes ~1 h on a laptop CPU)")
    args = parser.parse_args()

    embedder = SentenceTransformer(MODEL_NAME)
    df, embeddings = build_or_load(args.data_dir, args.cache_dir, embedder, args.limit)
    uvicorn.run(create_app(df, embeddings, embedder), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
