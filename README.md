# agent-19 · Learned feature reranker over BM25 (SciFact)

**Live demo:** https://agent-19-distilled-reranker.braylo777.workers.dev · API: `POST /api/rerank {"query": "...", "docs": [{"title","text"}]}`

## What, and an honest scope note
The plan was to distill a cross-encoder teacher into a small student on CPU. The box has about 2 GB of free disk, too little for PyTorch plus a cross-encoder. So **this release is a learned feature reranker, not a distilled one.** It is a 9-feature linear model trained with pairwise-logistic (RankNet-style) SGD on SciFact's gold *train* qrels, in plain JS on CPU. Distillation from a cross-encoder teacher (`@cf/baai/bge-reranker-base` on Workers AI) is the post-reset step below, and this repo already has the student, trainer and eval for it.

Pipeline: BM25 (k1 0.9, b 0.4) top-100 candidates → features (BM25 norm/log, IDF-weighted query coverage of body and title, bigram overlap, lead-60-token coverage, inverse rank, log length) → linear score → reranked top-10.

## Results (real, from `evals/results.json`, measured 2026-10-27, held-out 300 test queries)
| metric | value |
|---|---|
| BM25 nDCG@10 | **0.6602** |
| learned reranker nDCG@10 | **0.6692** |
| lift | **+0.009** (paired bootstrap 95% CI -0.0021 to +0.0199) |
| queries improved / hurt | 34 / 18 |
| BM25 recall@100 (reranker ceiling) | 0.8859 |
| rerank cost | 0.0066 ms per query (100 candidates, CPU) |
| training time | 0.1537 s CPU (feature extraction 10.3344 s) |

**The lift is small and not statistically significant (the CI crosses 0).** Lexical features alone barely beat a well-tuned BM25 on SciFact. That is the motivation for the teacher-distillation step. Our BM25 (0.660) matches the published BEIR SciFact BM25 baseline (0.665), which sanity-checks the eval.

Learned weights: bias 0, bm25_norm 1.5067, log_bm25 4.0172, idf_coverage 0.5326, title_idf_coverage 2.4299, bigram_overlap -0.4305, lead_coverage -0.1479, inv_rank 0.4003, log_len -3.6594.

## How to run
```bash
bash evals/get-data.sh   # downloads BEIR SciFact (~2.8 MB), not committed (CC BY-NC 2.0)
npm run eval             # BM25 + train + eval -> evals/results.json, src/weights.js, src/demo.js
npm test                 # 11 tests
npm run build            # dist/worker.js;  or: npx wrangler deploy
```
No dependencies. Free tier only. Rollback tag: `v0-rollback`. CI job in `ci/github-actions-ci.yml` (copy to `.github/workflows/` to enable).

## Pending (after the Workers AI quota reset, 7 PM CT Oct 28 2026)
- Teacher: score the top-20 BM25 candidates for the 809 train queries with `@cf/baai/bge-reranker-base` (about 16k pairs, spread over days to stay inside the free 10k-neuron allocation).
- Distill: retrain the student on teacher soft labels plus gold, then report student vs BM25 vs teacher nDCG@10 and latency.
- Add a dense feature (bge-small cosine) and measure its lift.
