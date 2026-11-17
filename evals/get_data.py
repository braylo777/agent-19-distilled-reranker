"""Downloads BEIR SciFact (~2.8 MB zip) into evals/data/. Run this yourself; the training eval needs it."""

import io, urllib.request, zipfile
from pathlib import Path

dest = Path(__file__).resolve().parent / "data"
dest.mkdir(exist_ok=True)
URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
with urllib.request.urlopen(URL, timeout=120) as r:
    zipfile.ZipFile(io.BytesIO(r.read())).extractall(dest)
print("saved to", dest / "scifact")
