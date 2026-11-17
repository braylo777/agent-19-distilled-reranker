"""Learned feature reranker (linear, pairwise-logistic) over BM25 candidates. Pure Python, CPU only."""

import json, math, re
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"
WEIGHTS = json.loads((DATA / "weights.json").read_text())
STOP = set(
    "a an the of in on and or to for with by is are was were be as at from that this these those it its we our which than".split(
        " "
    )
)
_TOK = re.compile(r"[a-z0-9]+")
FEATURE_NAMES = [
    "bias",
    "bm25_norm",
    "log_bm25",
    "idf_coverage",
    "title_idf_coverage",
    "bigram_overlap",
    "lead_coverage",
    "inv_rank",
    "log_len",
]


def tok(s):
    return [t for t in _TOK.findall(s.lower()) if t not in STOP]


class Index:
    def __init__(self, docs, k1=0.9, b=0.4):
        self.k1, self.b = k1, b
        self.D = []
        df = {}
        total = 0
        for d in docs:
            body = tok(d["title"] + " " + d["text"])
            tf = {}
            for t in body:
                tf[t] = tf.get(t, 0) + 1
            for t in tf:
                df[t] = df.get(t, 0) + 1
            total += len(body)
            self.D.append(
                {
                    "id": d["id"],
                    "title": tok(d["title"]),
                    "body": body,
                    "raw": d,
                    "tf": tf,
                }
            )
        self.N = len(self.D)
        self.avg = total / self.N if self.N else math.nan
        self.df = df
        self.post = {}
        for i, d in enumerate(self.D):
            for t in d["tf"]:
                self.post.setdefault(t, []).append(i)

    def idf(self, t):
        n = self.df.get(t, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def bm25(self, q, d):
        s = 0.0
        for t in dict.fromkeys(q):
            f = d["tf"].get(t)
            if f:
                s += (
                    self.idf(t)
                    * f
                    * (self.k1 + 1)
                    / (f + self.k1 * (1 - self.b + self.b * len(d["body"]) / self.avg))
                )
        return s

    def search(self, query, k=100):
        q = tok(query)
        cand = {}
        for t in q:
            for i in self.post.get(t, []):
                cand[i] = None
        hits = [{"i": i, "s": self.bm25(q, self.D[i])} for i in cand]
        hits.sort(key=lambda h: -h["s"])  # stable, like JS
        return hits[:k]


def build_index(docs, k1=0.9, b=0.4):
    return Index(docs, k1, b)


def _bigrams(a):
    return {a[i] + " " + a[i + 1] for i in range(len(a) - 1)}


def features(ix, query, d, rank, top):
    q, ts, bs = tok(query), set(d["title"]), d["tf"]
    iq = sum(ix.idf(t) for t in q) or 1
    cov = sum(ix.idf(t) for t in q if t in bs) / iq
    tcov = sum(ix.idf(t) for t in q if t in ts) / iq
    qb, db = _bigrams(q), _bigrams(d["body"])
    bg = sum(1 for x in qb if x in db) / len(qb) if qb else 0
    first = d["body"][:60]
    fcov = sum(1 for t in q if t in first) / (len(q) or 1)
    return [
        1,
        d["s"] / (top or 1),
        math.log1p(d["s"]),
        cov,
        tcov,
        bg,
        fcov,
        1 / (1 + rank),
        math.log1p(len(d["body"])) / 6,
    ]


def score(w, f):
    s = 0.0
    for x, wi in zip(f, w):
        s += x * wi
    return s


def train(groups, epochs=30, lr=0.05, l2=1e-4, seed=1):
    """Pairwise logistic (RankNet-style) SGD on (pos, neg) pairs from the same query."""
    n = len(groups[0]["feats"][0])
    w = [0.0] * n
    w[1] = 1
    s = [seed & 0xFFFFFFFF]

    def rnd():
        s[0] = (s[0] * 1664525 + 1013904223) & 0xFFFFFFFF
        return s[0] / 2**32

    for _ in range(epochs):
        for g in groups:
            P = [i for i, r in enumerate(g["rel"]) if r > 0]
            N = [i for i, r in enumerate(g["rel"]) if not r > 0]
            if not P or not N:
                continue
            for _k in range(10):
                a = g["feats"][P[math.floor(rnd() * len(P))]]
                b = g["feats"][N[math.floor(rnd() * len(N))]]
                z = score(w, a) - score(w, b)
                gr = 1 / (1 + math.exp(z))
                for j in range(n):
                    w[j] += lr * (gr * (a[j] - b[j]) - l2 * w[j])
    return w


def ndcg(ranked_rel, all_rel, k=10):
    dcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ranked_rel[:k]))
    ideal = sum(
        (2**r - 1) / math.log2(i + 2)
        for i, r in enumerate(sorted(all_rel, reverse=True)[:k])
    )
    return dcg / ideal if ideal else 0


def _js_str(v):
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def rerank(query, docs):
    ix = build_index(
        [
            {
                "id": _js_str(d["id"]) if d.get("id") is not None else str(i),
                "title": str(d.get("title") or ""),
                "text": str(d.get("text") or ""),
            }
            for i, d in enumerate(docs)
        ]
    )
    hits = ix.search(query, 100)
    top = (hits[0]["s"] if hits else 0) or 1
    out = []
    for r, h in enumerate(hits):
        d = {**ix.D[h["i"]], "s": h["s"]}
        out.append(
            {
                "id": d["id"],
                "title": d["raw"]["title"],
                "bm25": h["s"],
                "bm25_rank": r + 1,
                "score": score(WEIGHTS, features(ix, query, d, r, top)),
            }
        )
    out.sort(key=lambda x: -x["score"])
    return out


def handle(method, path, body_text=""):
    """Pure router: (status, json); status 0 = serve the page."""
    if path == "/api/rerank" and method == "POST":
        try:
            b = json.loads(body_text)
        except ValueError:
            return 400, {"error": "bad json"}
        q = b.get("query") if isinstance(b, dict) else None
        docs = b.get("docs") if isinstance(b, dict) else None
        if not q or not isinstance(docs, list) or len(docs) > 200:
            return 400, {"error": "need query and 1-200 docs"}
        return 200, rerank(str(q)[:1000], docs[:200])
    if path == "/health":
        return 200, {"ok": True}
    return 0, None
