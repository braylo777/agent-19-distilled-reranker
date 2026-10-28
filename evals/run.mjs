import fs from "node:fs"; import { buildIndex, features, train, score, ndcg, FEATURE_NAMES } from "../src/core.js";
const dir = new URL("./data/scifact/", import.meta.url); const jl = (f) => fs.readFileSync(new URL(f, dir), "utf8").trim().split("\n").map((l) => JSON.parse(l));
const docs = jl("corpus.jsonl").map((d) => ({ id: d._id, title: d.title, text: d.text })); const Q = new Map(jl("queries.jsonl").map((q) => [q._id, q.text]));
const qrels = (f) => { const m = new Map(); for (const l of fs.readFileSync(new URL("qrels/" + f, dir), "utf8").trim().split("\n").slice(1)) { const [q, d, r] = l.split("\t"); if (!m.has(q)) m.set(q, new Map()); m.get(q).set(d, +r); } return m; };
const K = 100, ix = buildIndex(docs);
const groupsFor = (qr) => [...qr].map(([qid, rel]) => { const hits = ix.search(Q.get(qid), K); const top = hits[0]?.s || 1; const cand = hits.map((h, r) => ({ ...ix.D[h.i], s: h.s, r }));
  return { qid, ids: cand.map((c) => c.id), feats: cand.map((c) => features(ix, Q.get(qid), c, c.r, top)), rel: cand.map((c) => rel.get(c.id) || 0), allRel: [...rel.values()], recall: cand.filter((c) => rel.has(c.id)).length / rel.size }; });
const t0 = performance.now(); const tr = groupsFor(qrels("train.tsv")), te = groupsFor(qrels("test.tsv")); const featMs = performance.now() - t0;
const t1 = performance.now(); const w = train(tr); const trainMs = performance.now() - t1;
const evalG = (G, f) => G.reduce((s, g) => { const o = g.rel.map((r, i) => [f(g, i), r]).sort((a, b) => b[0] - a[0]).map((x) => x[1]); return s + ndcg(o, g.allRel); }, 0) / G.length;
const bm = (g, i) => -i, rr = (g, i) => score(w, g.feats[i]); const r4 = (x) => Math.round(x * 1e4) / 1e4;
// paired bootstrap for lift CI
const per = te.map((g) => [evalG([g], rr) - evalG([g], bm)]); let s = 7; const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32); const boots = [];
for (let b = 0; b < 1000; b++) { let t = 0; for (let i = 0; i < per.length; i++) t += per[Math.floor(rnd() * per.length)][0]; boots.push(t / per.length); } boots.sort((a, b) => a - b);
const t2 = performance.now(); for (const g of te) g.feats.map((f) => score(w, f)); const rerankMsPerQ = (performance.now() - t2) / te.length;
const res = { dataset: "BEIR SciFact (5,183 docs; 809 train / 300 test queries)", candidates: K, test: { bm25_ndcg10: r4(evalG(te, bm)), reranker_ndcg10: r4(evalG(te, rr)), lift: r4(evalG(te, rr) - evalG(te, bm)), lift_95ci: [r4(boots[25]), r4(boots[975])], bm25_recall_at_100: r4(te.reduce((s, g) => s + g.recall, 0) / te.length), queries_improved: per.filter((x) => x[0] > 1e-9).length, queries_hurt: per.filter((x) => x[0] < -1e-9).length },
  train: { bm25_ndcg10: r4(evalG(tr, bm)), reranker_ndcg10: r4(evalG(tr, rr)) }, weights: Object.fromEntries(FEATURE_NAMES.map((n, i) => [n, r4(w[i])])), cpu: { feature_extraction_s: r4(featMs / 1000), train_s: r4(trainMs / 1000), rerank_ms_per_query_100_cands: r4(rerankMsPerQ) } };
fs.writeFileSync(new URL("./results.json", import.meta.url), JSON.stringify(res, null, 1)); fs.writeFileSync(new URL("../src/weights.js", import.meta.url), `export const WEIGHTS = ${JSON.stringify(w.map(r4))};\n`);
// demo payload: 40 test queries, top-10 BM25 vs reranked, titles + relevance
const demo = te.slice(0, 40).map((g) => { const ord = g.rel.map((r, i) => i).sort((a, b) => rr(g, b) - rr(g, a)); const t = (i) => ({ id: g.ids[i], title: docs.find((d) => d.id === g.ids[i]).title.slice(0, 140), rel: g.rel[i] });
  return { q: Q.get(g.qid), bm25: g.ids.slice(0, 10).map((_, i) => t(i)), rerank: ord.slice(0, 10).map(t) }; }); fs.writeFileSync(new URL("../src/demo.js", import.meta.url), `export const DEMO = ${JSON.stringify(demo)};\n`);
console.log(JSON.stringify(res, null, 1));
