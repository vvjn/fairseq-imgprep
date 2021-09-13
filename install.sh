#!/bin/bash
set -euxo pipefail

cd external/apex && pip install . && cd ../..

cd external/bottom-up-attention.pytorch/detectron2 && pip install . && cd ../../..

cd external/bottom-up-attention.pytorch && pip install -e . && cd ../..

cd external/scene_graph_benchmark && pip install -e . && cd ../..
