// Bundles src/*.js into dist/worker.js for API upload.
import fs from "node:fs"; const strip = (f) => fs.readFileSync("src/" + f, "utf8").replace(/^import .*\n/gm, "").replace(/^export (const|function|class)/gm, "$1");
fs.mkdirSync("dist", { recursive: true }); fs.writeFileSync("dist/worker.js", ["core.js", "weights.js", "demo.js", "index.js"].map(strip).join("\n"));
