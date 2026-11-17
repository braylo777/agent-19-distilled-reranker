import json
from pathlib import Path

from agent19.core import (
    FEATURE_NAMES,
    WEIGHTS,
    build_index,
    features,
    handle,
    ndcg,
    rerank,
    score,
    tok,
    train,
)

docs = [
    {
        "id": "1",
        "title": "Aspirin reduces heart attack risk",
        "text": "Low dose aspirin lowers myocardial infarction.",
    },
    {"id": "2", "title": "Coffee and sleep", "text": "Caffeine delays sleep onset."},
    {
        "id": "3",
        "title": "Heart rate in athletes",
        "text": "Athletes have low resting heart rate.",
    },
]


def test_tokenizer_and_bm25():
    assert tok("The heart of it") == ["heart"]
    ix = build_index(docs)
    assert ix.D[ix.search("aspirin heart attack")[0]["i"]]["id"] == "1"
    assert len(build_index(docs).search("caffeine")) == 1


def test_features_and_ndcg():
    ix = build_index(docs)
    d = {**ix.D[0], "s": 1}
    assert len(features(ix, "aspirin", d, 0, 1)) == len(FEATURE_NAMES)
    assert ndcg([1, 0, 0], [1]) == 1 and ndcg([0, 1], [1]) < 1 and ndcg([0, 0], []) == 0


def test_trainer_and_weights():
    g = {"feats": [[1, 1, 0], [1, 0, 1]], "rel": [1, 0]}
    w = train([g], epochs=50)
    assert score(w, g["feats"][0]) > score(w, g["feats"][1])
    assert len(WEIGHTS) == len(FEATURE_NAMES)


def test_rerank_api_and_results():
    assert len(rerank("heart", docs)) == 2
    assert (
        handle("POST", "/api/rerank", "{nope")[0] == 400
        and handle("POST", "/api/rerank", json.dumps({"query": "x"}))[0] == 400
    )
    assert (
        handle("POST", "/api/rerank", json.dumps({"query": "heart", "docs": docs}))[0]
        == 200
        and handle("GET", "/health")[1] == {"ok": True}
        and handle("GET", "/")[0] == 0
    )
    r = json.loads(
        (Path(__file__).resolve().parents[1] / "evals" / "results.json").read_text()
    )
    assert r["test"]["reranker_ndcg10"] >= r["test"]["bm25_ndcg10"]
