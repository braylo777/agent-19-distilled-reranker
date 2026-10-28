#!/usr/bin/env bash
# Downloads BEIR SciFact (~2.8 MB zip) into evals/data/
cd "$(dirname "$0")/data" && curl -sL -o s.zip https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip && unzip -qo s.zip && rm s.zip
