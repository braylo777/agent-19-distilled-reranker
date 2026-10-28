// Learned feature reranker (linear, pairwise-logistic) over BM25 candidates. Pure JS, CPU only.
export const tok = (s) => (s.toLowerCase().match(/[a-z0-9]+/g) || []).filter((t) => !STOP.has(t));
const STOP = new Set("a an the of in on and or to for with by is are was were be as at from that this these those it its we our which than".split(" "));
export function buildIndex(docs, k1 = 0.9, b = 0.4) { // docs: [{id,title,text}]
  const D = docs.map((d) => ({ id: d.id, title: tok(d.title), body: tok(d.title + " " + d.text), raw: d })); const df = new Map(); let L = 0;
  for (const d of D) { L += d.body.length; d.tf = new Map(); for (const t of d.body) d.tf.set(t, (d.tf.get(t) || 0) + 1); for (const t of d.tf.keys()) df.set(t, (df.get(t) || 0) + 1); }
  const N = D.length, avg = L / N, idf = (t) => Math.log(1 + (N - (df.get(t) || 0) + 0.5) / ((df.get(t) || 0) + 0.5));
  const post = new Map(); D.forEach((d, i) => { for (const t of d.tf.keys()) { if (!post.has(t)) post.set(t, []); post.get(t).push(i); } });
  const bm25 = (q, d) => { let s = 0; for (const t of new Set(q)) { const f = d.tf.get(t); if (f) s += idf(t) * f * (k1 + 1) / (f + k1 * (1 - b + b * d.body.length / avg)); } return s; };
  const search = (query, k = 100) => { const q = tok(query); const cand = new Set(); for (const t of q) for (const i of post.get(t) || []) cand.add(i);
    return [...cand].map((i) => ({ i, s: bm25(q, D[i]) })).sort((x, y) => y.s - x.s).slice(0, k); };
  return { D, idf, bm25, search };
}
const bigrams = (a) => new Set(a.slice(1).map((t, i) => a[i] + " " + t));
export function features(ix, query, d, rank, top) { const q = tok(query), qs = new Set(q), ts = new Set(d.title), bs = d.tf; const iq = q.reduce((s, t) => s + ix.idf(t), 0) || 1;
  const cov = q.filter((t) => bs.has(t)).reduce((s, t) => s + ix.idf(t), 0) / iq, tcov = q.filter((t) => ts.has(t)).reduce((s, t) => s + ix.idf(t), 0) / iq;
  const qb = bigrams(q), db = bigrams(d.body); const bg = qb.size ? [...qb].filter((x) => db.has(x)).length / qb.size : 0;
  const first = d.body.slice(0, 60), fcov = q.filter((t) => first.includes(t)).length / (q.length || 1);
  return [1, d.s / (top || 1), Math.log1p(d.s), cov, tcov, bg, fcov, 1 / (1 + rank), Math.log1p(d.body.length) / 6]; }
export const FEATURE_NAMES = ["bias", "bm25_norm", "log_bm25", "idf_coverage", "title_idf_coverage", "bigram_overlap", "lead_coverage", "inv_rank", "log_len"];
export const score = (w, f) => f.reduce((s, x, i) => s + x * w[i], 0);
// Pairwise logistic (RankNet-style) SGD on (pos, neg) pairs from the same query.
export function train(groups, { epochs = 30, lr = 0.05, l2 = 1e-4, seed = 1 } = {}) { const n = groups[0].feats[0].length; const w = new Array(n).fill(0); w[1] = 1; let s = seed;
  const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32);
  for (let e = 0; e < epochs; e++) for (const g of groups) { const P = g.rel.map((r, i) => r > 0 ? i : -1).filter((i) => i >= 0), Nn = g.rel.map((r, i) => r > 0 ? -1 : i).filter((i) => i >= 0); if (!P.length || !Nn.length) continue;
    for (let k = 0; k < 10; k++) { const a = g.feats[P[Math.floor(rnd() * P.length)]], b = g.feats[Nn[Math.floor(rnd() * Nn.length)]]; const z = score(w, a) - score(w, b), gr = 1 / (1 + Math.exp(z));
      for (let j = 0; j < n; j++) w[j] += lr * (gr * (a[j] - b[j]) - l2 * w[j]); } } return w; }
export function ndcg(rankedRel, allRel, k = 10) { const dcg = rankedRel.slice(0, k).reduce((s, r, i) => s + (2 ** r - 1) / Math.log2(i + 2), 0);
  const ideal = [...allRel].sort((a, b) => b - a).slice(0, k).reduce((s, r, i) => s + (2 ** r - 1) / Math.log2(i + 2), 0); return ideal ? dcg / ideal : 0; }
