# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

python -m venv .venv/matmul_bench
source .venv/matmul_bench/bin/activate
pip install wheel
cd reptil; pip install .; cd ..
ninja
