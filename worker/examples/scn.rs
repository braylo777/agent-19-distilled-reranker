// Parity helper: same corpus/queries as the legacy JS harness; prints JSON.
use agent_19_distilled_reranker::core::*;
use serde_json::{json, Value};
fn main() {
    let d: Value = serde_json::from_str(&std::fs::read_to_string("/tmp/in19.json").unwrap()).unwrap();
    let docs: Vec<Doc> = d["docs"].as_array().unwrap().iter().map(|x| Doc { id: x["id"].as_str().unwrap().into(), title: x["title"].as_str().unwrap().into(), text: x["text"].as_str().unwrap().into() }).collect();
    let out: Vec<Value> = d["qs"].as_array().unwrap().iter().map(|q| Value::Array(rerank(q.as_str().unwrap(), &docs).into_iter().take(30).map(|r| json!([r["id"], r["bm25_rank"], r["bm25"], r["score"]])).collect())).collect();
    let g = vec![(vec![vec![1.0, 0.9, 0.2], vec![1.0, 0.1, 0.7], vec![1.0, 0.5, 0.5]], vec![1.0, 0.0, 0.0]), (vec![vec![1.0, 0.2, 0.8], vec![1.0, 0.6, 0.1]], vec![0.0, 2.0]), (vec![vec![1.0, 1.0, 1.0]], vec![0.0])];
    println!("{}", serde_json::to_string(&json!({"out": out, "w": train(&g, 40, 0.05, 1e-4, 1), "w2": train(&g, 5, 0.1, 1e-4, 9), "nd": [ndcg(&[3.0, 2.0, 0.0, 1.0], &[3.0, 2.0, 1.0, 0.0], 10), ndcg(&[0.0, 0.0], &[1.0], 10), ndcg(&[1.0], &[], 10)]})).unwrap());
}
