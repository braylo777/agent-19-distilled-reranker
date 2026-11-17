//! Cloudflare Worker entry (wasm32 only). Same routes and JSON as the old src/index.js.
use crate::core::handle;
use crate::page::page;
use worker::*;

#[event(fetch)]
async fn fetch(mut req: Request, _env: Env, _ctx: Context) -> Result<Response> {
    let path = req.url()?.path().to_string();
    let method = req.method();
    let body = if method == Method::Post { req.text().await.unwrap_or_default() } else { String::new() };
    let (status, v) = handle(method.as_ref(), &path, &body);
    if status == 0 {
        let h = Headers::new();
        h.set("content-type", "text/html; charset=utf-8")?;
        return Ok(Response::ok(page())?.with_headers(h));
    }
    Ok(Response::from_json(&v)?.with_status(status))
}
