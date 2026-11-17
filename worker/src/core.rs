//! Learned feature reranker (linear, pairwise-logistic) over BM25 candidates. CPU only.
use regex::Regex;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

const WEIGHTS_JSON: &str = include_str!("../data/weights.json");
pub const FEATURE_NAMES: [&str; 9] = [
    "bias",
    "bm25_norm",
    "log_bm25",
    "idf_coverage",
    "title_idf_coverage",
    "bigram_overlap",
    "lead_coverage",
    "inv_rank",
    "log_len",
];
const STOP: &str = "a an the of in on and or to for with by is are was were be as at from that this these those it its we our which than";

pub fn weights() -> Vec<f64> {
    serde_json::from_str(WEIGHTS_JSON).unwrap()
}

pub fn tok(s: &str) -> Vec<String> {
    static R: OnceLock<Regex> = OnceLock::new();
    static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    let stop = S.get_or_init(|| STOP.split(' ').collect());
    R.get_or_init(|| Regex::new("[a-z0-9]+").unwrap())
        .find_iter(&s.to_lowercase())
        .map(|m| m.as_str())
        .filter(|t| !stop.contains(t))
        .map(String::from)
        .collect()
}

pub struct Doc {
    pub id: String,
    pub title: String,
    pub text: String,
}
pub struct Item {
    pub id: String,
    pub raw_title: String,
    pub title: HashSet<String>,
    pub body: Vec<String>,
    pub tf: HashMap<String, f64>,
}
pub struct Index {
    pub d: Vec<Item>,
    df: HashMap<String, f64>,
    n: f64,
    avg: f64,
    post: HashMap<String, Vec<usize>>,
    k1: f64,
    b: f64,
}

impl Index {
    pub fn new(docs: &[Doc]) -> Index {
        let (k1, b) = (0.9, 0.4);
        let mut d = vec![];
        let (mut df, mut total): (HashMap<String, f64>, usize) = (HashMap::new(), 0);
        for x in docs {
            let body = tok(&format!("{} {}", x.title, x.text));
            let mut tf: HashMap<String, f64> = HashMap::new();
            for t in &body {
                *tf.entry(t.clone()).or_insert(0.0) += 1.0;
            }
            for t in tf.keys() {
                *df.entry(t.clone()).or_insert(0.0) += 1.0;
            }
            total += body.len();
            d.push(Item {
                id: x.id.clone(),
                raw_title: x.title.clone(),
                title: tok(&x.title).into_iter().collect(),
                body,
                tf,
            });
        }
        let n = d.len() as f64;
        let mut post: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, it) in d.iter().enumerate() {
            // postings in document order; per-term order within a doc does not matter
            for t in it.tf.keys() {
                post.entry(t.clone()).or_default().push(i);
            }
        }
        Index {
            d,
            df,
            n,
            avg: total as f64 / n,
            post,
            k1,
            b,
        }
    }
    pub fn idf(&self, t: &str) -> f64 {
        let n = *self.df.get(t).unwrap_or(&0.0);
        (1.0 + (self.n - n + 0.5) / (n + 0.5)).ln()
    }
    fn bm25(&self, q: &[String], it: &Item) -> f64 {
        let mut seen = HashSet::new();
        let mut s = 0.0;
        for t in q {
            if !seen.insert(t.as_str()) {
                continue;
            }
            if let Some(&f) = it.tf.get(t) {
                s += self.idf(t) * f * (self.k1 + 1.0)
                    / (f + self.k1 * (1.0 - self.b + self.b * it.body.len() as f64 / self.avg));
            }
        }
        s
    }
    /// (index, bm25 score) for the top k candidates.
    pub fn search(&self, query: &str, k: usize) -> Vec<(usize, f64)> {
        let q = tok(query);
        let mut cand: Vec<usize> = vec![];
        let mut seen = HashSet::new();
        for t in &q {
            for &i in self.post.get(t).map(|v| v.as_slice()).unwrap_or(&[]) {
                if seen.insert(i) {
                    cand.push(i);
                }
            }
        }
        let mut hits: Vec<(usize, f64)> = cand
            .into_iter()
            .map(|i| (i, self.bm25(&q, &self.d[i])))
            .collect();
        hits.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal)); // stable
        hits.truncate(k);
        hits
    }
}

fn bigrams(a: &[String]) -> HashSet<String> {
    a.windows(2).map(|w| format!("{} {}", w[0], w[1])).collect()
}

pub fn features(ix: &Index, query: &str, it: &Item, s: f64, rank: usize, top: f64) -> Vec<f64> {
    let q = tok(query);
    let iq = {
        let v: f64 = q.iter().map(|t| ix.idf(t)).sum();
        if v == 0.0 {
            1.0
        } else {
            v
        }
    };
    let cov = q
        .iter()
        .filter(|t| it.tf.contains_key(*t))
        .map(|t| ix.idf(t))
        .sum::<f64>()
        / iq;
    let tcov = q
        .iter()
        .filter(|t| it.title.contains(*t))
        .map(|t| ix.idf(t))
        .sum::<f64>()
        / iq;
    let (qb, db) = (bigrams(&q), bigrams(&it.body));
    let bg = if qb.is_empty() {
        0.0
    } else {
        qb.iter().filter(|x| db.contains(*x)).count() as f64 / qb.len() as f64
    };
    let first = &it.body[..it.body.len().min(60)];
    let fcov = q.iter().filter(|t| first.contains(t)).count() as f64
        / if q.is_empty() { 1.0 } else { q.len() as f64 };
    vec![
        1.0,
        s / if top == 0.0 { 1.0 } else { top },
        s.ln_1p(),
        cov,
        tcov,
        bg,
        fcov,
        1.0 / (1.0 + rank as f64),
        (it.body.len() as f64).ln_1p() / 6.0,
    ]
}

pub fn score(w: &[f64], f: &[f64]) -> f64 {
    let mut s = 0.0;
    for (x, wi) in f.iter().zip(w) {
        s += x * wi;
    }
    s
}

/// Pairwise logistic (RankNet-style) SGD on (pos, neg) pairs from the same query. groups: (features, relevance).
pub fn train(
    groups: &[(Vec<Vec<f64>>, Vec<f64>)],
    epochs: usize,
    lr: f64,
    l2: f64,
    seed: u32,
) -> Vec<f64> {
    let n = groups[0].0[0].len();
    let mut w = vec![0.0; n];
    w[1] = 1.0;
    let mut s = seed;
    let mut rnd = move || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        s as f64 / 4294967296.0
    };
    for _ in 0..epochs {
        for (feats, rel) in groups {
            let p: Vec<usize> = rel
                .iter()
                .enumerate()
                .filter(|(_, r)| **r > 0.0)
                .map(|(i, _)| i)
                .collect();
            let ng: Vec<usize> = rel
                .iter()
                .enumerate()
                .filter(|(_, r)| !(**r > 0.0))
                .map(|(i, _)| i)
                .collect();
            if p.is_empty() || ng.is_empty() {
                continue;
            }
            for _ in 0..10 {
                let a = &feats[p[(rnd() * p.len() as f64).floor() as usize]];
                let b = &feats[ng[(rnd() * ng.len() as f64).floor() as usize]];
                let z = score(&w, a) - score(&w, b);
                let gr = 1.0 / (1.0 + z.exp());
                for j in 0..n {
                    w[j] += lr * (gr * (a[j] - b[j]) - l2 * w[j]);
                }
            }
        }
    }
    w
}

pub fn ndcg(ranked_rel: &[f64], all_rel: &[f64], k: usize) -> f64 {
    let dcg: f64 = ranked_rel
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, r)| (2f64.powf(*r) - 1.0) / ((i + 2) as f64).log2())
        .sum();
    let mut sorted = all_rel.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let ideal: f64 = sorted
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, r)| (2f64.powf(*r) - 1.0) / ((i + 2) as f64).log2())
        .sum();
    if ideal != 0.0 {
        dcg / ideal
    } else {
        0.0
    }
}

pub fn rerank(query: &str, docs: &[Doc]) -> Vec<Value> {
    let ix = Index::new(docs);
    let hits = ix.search(query, 100);
    let top = hits.first().map_or(0.0, |h| h.1);
    let top = if top == 0.0 { 1.0 } else { top };
    let w = weights();
    let mut out: Vec<(f64, Value)> = hits
        .iter()
        .enumerate()
        .map(|(r, (i, s))| {
            let it = &ix.d[*i];
            let sc = score(&w, &features(&ix, query, it, *s, r, top));
            (sc, json!({"id": it.id, "title": it.raw_title, "bm25": s, "bm25_rank": r + 1, "score": sc}))
        })
        .collect();
    out.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    out.into_iter().map(|x| x.1).collect()
}

fn js_str(v: Option<&Value>, default: String) -> String {
    match v {
        None | Some(Value::Null) => default,
        Some(Value::String(s)) => s.clone(),
        Some(Value::Number(n)) => n
            .as_f64()
            .filter(|f| f.fract() == 0.0)
            .map_or(n.to_string(), |f| format!("{}", f as i64)),
        Some(o) => o.to_string(),
    }
}

/// Pure router: (status, json); status 0 = serve the page.
pub fn handle(method: &str, path: &str, body_text: &str) -> (u16, Value) {
    if path == "/api/rerank" && method == "POST" {
        let Ok(b) = serde_json::from_str::<Value>(body_text) else {
            return (400, json!({"error": "bad json"}));
        };
        let q = b
            .get("query")
            .filter(|q| !q.is_null() && **q != json!(false) && **q != json!(0) && **q != json!(""));
        let docs = b.get("docs").and_then(|d| d.as_array());
        let (Some(q), Some(docs)) = (q, docs) else {
            return (400, json!({"error": "need query and 1-200 docs"}));
        };
        if docs.len() > 200 {
            return (400, json!({"error": "need query and 1-200 docs"}));
        }
        let qs = match q {
            Value::String(s) => s.clone(),
            o => o.to_string(),
        };
        let qs: String =
            String::from_utf16_lossy(&qs.encode_utf16().take(1000).collect::<Vec<u16>>());
        let ds: Vec<Doc> = docs
            .iter()
            .enumerate()
            .map(|(i, d)| Doc {
                id: js_str(d.get("id"), i.to_string()),
                title: js_str(
                    d.get("title").filter(|t| !t.is_null() && **t != json!("")),
                    String::new(),
                ),
                text: js_str(
                    d.get("text").filter(|t| !t.is_null() && **t != json!("")),
                    String::new(),
                ),
            })
            .collect();
        return (200, Value::Array(rerank(&qs, &ds)));
    }
    if path == "/health" {
        return (200, json!({"ok": true}));
    }
    (0, Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn docs() -> Vec<Doc> {
        [
            (
                "1",
                "Aspirin reduces heart attack risk",
                "Low dose aspirin lowers myocardial infarction.",
            ),
            ("2", "Coffee and sleep", "Caffeine delays sleep onset."),
            (
                "3",
                "Heart rate in athletes",
                "Athletes have low resting heart rate.",
            ),
        ]
        .iter()
        .map(|(i, t, x)| Doc {
            id: i.to_string(),
            title: t.to_string(),
            text: x.to_string(),
        })
        .collect()
    }
    #[test]
    fn tokenizer_bm25_features_ndcg() {
        assert_eq!(tok("The heart of it"), ["heart"]);
        let ix = Index::new(&docs());
        assert_eq!(ix.d[ix.search("aspirin heart attack", 100)[0].0].id, "1");
        assert_eq!(Index::new(&docs()).search("caffeine", 100).len(), 1);
        let it = &ix.d[0];
        assert_eq!(
            features(&ix, "aspirin", it, 1.0, 0, 1.0).len(),
            FEATURE_NAMES.len()
        );
        assert_eq!(ndcg(&[1.0, 0.0, 0.0], &[1.0], 10), 1.0);
        assert!(ndcg(&[0.0, 1.0], &[1.0], 10) < 1.0);
        assert_eq!(ndcg(&[0.0, 0.0], &[], 10), 0.0);
    }
    #[test]
    fn trainer_weights_and_api() {
        let g = vec![(
            vec![vec![1.0, 1.0, 0.0], vec![1.0, 0.0, 1.0]],
            vec![1.0, 0.0],
        )];
        let w = train(&g, 50, 0.05, 1e-4, 1);
        assert!(score(&w, &g[0].0[0]) > score(&w, &g[0].0[1]));
        assert_eq!(weights().len(), FEATURE_NAMES.len());
        assert_eq!(rerank("heart", &docs()).len(), 2);
        assert_eq!(handle("POST", "/api/rerank", "{nope").0, 400);
        assert_eq!(
            handle("POST", "/api/rerank", &json!({"query": "x"}).to_string()).0,
            400
        );
        assert_eq!(
            handle(
                "POST",
                "/api/rerank",
                &json!({"query": "heart", "docs": [{"title": "Heart rate", "text": "x"}]})
                    .to_string()
            )
            .0,
            200
        );
        assert_eq!(handle("GET", "/health", "").1, json!({"ok": true}));
        assert_eq!(handle("GET", "/", "").0, 0);
    }
}
