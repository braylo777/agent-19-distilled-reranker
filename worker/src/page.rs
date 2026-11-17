//! Demo page: the old template literal, with the demo payload filled in at runtime.
const PRE: &str = r####"<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width"><title>agent-19 reranker</title><body style="font:14px system-ui;max-width:980px;margin:2em auto;padding:0 1em"><h1>agent-19 · Learned reranker vs BM25 on SciFact</h1><p>Pick a held-out SciFact test query. Green = relevant (gold qrels). <a href="https://github.com/braylo777/agent-19-distilled-reranker">GitHub</a> · API: POST /api/rerank {query, docs:[{title,text}]}</p><select id=s style="width:100%"></select><div style="display:flex;gap:1em"><ol id=a style="flex:1"></ol><ol id=b style="flex:1"></ol></div><script>const D="####;
const POST: &str = r####";D.forEach((d,i)=>s.add(new Option(d.q,i)));const li=x=>'<li style="color:'+(x.rel?'green;font-weight:bold':'#333')+'">'+x.title.replace(/</g,'&lt;')+'</li>';function show(){const d=D[s.value];a.innerHTML='<b>BM25</b>'+d.bm25.map(li).join('');b.innerHTML='<b>Reranked</b>'+d.rerank.map(li).join('')}s.onchange=show;show()</script>"####;
const DEMO_JSON: &str = include_str!("../data/demo.json");

pub fn page() -> String {
    format!("{}{}{}", PRE, DEMO_JSON.replace('<', "\\u003c"), POST)
}
