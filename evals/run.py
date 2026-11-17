"""Train and evaluate the learned reranker on BEIR SciFact (download it first: python3 evals/get_data.py).
Writes evals/results.json, data/weights.json and data/demo.json. Usage: python3 evals/run.py [--data DIR] [--out DIR]"""

import json, math, sys, time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "src"))
from agent19.core import (
    FEATURE_NAMES,
    build_index,
    features,
    ndcg,
    score,
    train,
)  # noqa: E402

arg = lambda k, d: Path(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
data = arg("--data", root / "evals" / "data" / "scifact")
out_dir = arg("--out", root)
jl = lambda f: [
    json.loads(l) for l in (data / f).read_text(encoding="utf-8").strip().split("\n")
]
docs = [
    {"id": d["_id"], "title": d["title"], "text": d["text"]} for d in jl("corpus.jsonl")
]
Q = {q["_id"]: q["text"] for q in jl("queries.jsonl")}


def qrels(f):
    m = {}
    for l in (data / "qrels" / f).read_text().strip().split("\n")[1:]:
        q, d, r = l.split("\t")
        m.setdefault(q, {})[d] = int(r)
    return m


K = 100
ix = build_index(docs)
r4 = lambda x: math.floor(x * 1e4 + 0.5) / 1e4


def groups_for(qr):
    out = []
    for qid, rel in qr.items():
        hits = ix.search(Q[qid], K)
        top = (hits[0]["s"] if hits else 0) or 1
        cand = [{**ix.D[h["i"]], "s": h["s"], "r": r} for r, h in enumerate(hits)]
        out.append(
            {
                "qid": qid,
                "ids": [c["id"] for c in cand],
                "feats": [features(ix, Q[qid], c, c["r"], top) for c in cand],
                "rel": [rel.get(c["id"], 0) for c in cand],
                "allRel": list(rel.values()),
                "recall": sum(1 for c in cand if c["id"] in rel) / len(rel),
            }
        )
    return out


t0 = time.perf_counter()
tr = groups_for(qrels("train.tsv"))
te = groups_for(qrels("test.tsv"))
feat_ms = (time.perf_counter() - t0) * 1000
t1 = time.perf_counter()
w = train(tr)
train_ms = (time.perf_counter() - t1) * 1000


def eval_g(G, f):
    tot = 0.0
    for g in G:
        o = [
            x[1]
            for x in sorted(
                ([f(g, i), r] for i, r in enumerate(g["rel"])), key=lambda x: -x[0]
            )
        ]
        tot += ndcg(o, g["allRel"])
    return tot / len(G)


bm = lambda g, i: -i
rr = lambda g, i: score(w, g["feats"][i])
per = [eval_g([g], rr) - eval_g([g], bm) for g in te]  # paired bootstrap for lift CI
s = [7]


def rnd():
    s[0] = (s[0] * 1664525 + 1013904223) & 0xFFFFFFFF
    return s[0] / 2**32


boots = sorted(
    sum(per[math.floor(rnd() * len(per))] for _ in range(len(per))) / len(per)
    for _ in range(1000)
)
t2 = time.perf_counter()
for g in te:
    [score(w, f) for f in g["feats"]]
rerank_ms = (time.perf_counter() - t2) * 1000 / len(te)
res = {
    "dataset": "BEIR SciFact (5,183 docs; 809 train / 300 test queries)",
    "candidates": K,
    "test": {
        "bm25_ndcg10": r4(eval_g(te, bm)),
        "reranker_ndcg10": r4(eval_g(te, rr)),
        "lift": r4(eval_g(te, rr) - eval_g(te, bm)),
        "lift_95ci": [r4(boots[25]), r4(boots[975])],
        "bm25_recall_at_100": r4(sum(g["recall"] for g in te) / len(te)),
        "queries_improved": sum(1 for x in per if x > 1e-9),
        "queries_hurt": sum(1 for x in per if x < -1e-9),
    },
    "train": {"bm25_ndcg10": r4(eval_g(tr, bm)), "reranker_ndcg10": r4(eval_g(tr, rr))},
    "weights": {n: r4(w[i]) for i, n in enumerate(FEATURE_NAMES)},
    "cpu": {
        "feature_extraction_s": r4(feat_ms / 1000),
        "train_s": r4(train_ms / 1000),
        "rerank_ms_per_query_100_cands": r4(rerank_ms),
    },
}
(out_dir / "evals").mkdir(exist_ok=True)
(out_dir / "data").mkdir(exist_ok=True)
(out_dir / "evals" / "results.json").write_text(json.dumps(res, indent=1))
(out_dir / "data" / "weights.json").write_text(json.dumps([r4(x) for x in w]))
by_id = {d["id"]: d for d in docs}


def item(g, i):
    return {
        "id": g["ids"][i],
        "title": by_id[g["ids"][i]]["title"][:140],
        "rel": g["rel"][i],
    }


demo = []
for g in te[
    :40
]:  # demo payload: 40 test queries, top-10 BM25 vs reranked, titles + relevance
    ordr = sorted(range(len(g["rel"])), key=lambda i: -rr(g, i))
    demo.append(
        {
            "q": Q[g["qid"]],
            "bm25": [item(g, i) for i in range(min(10, len(g["ids"])))],
            "rerank": [item(g, i) for i in ordr[:10]],
        }
    )
(out_dir / "data" / "demo.json").write_text(
    json.dumps(demo, separators=(",", ":"), ensure_ascii=False)
)
print(json.dumps(res, indent=1))
