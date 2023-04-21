python -m venv ~/envs/matmul_bench
source ~/envs/matmul_bench/bin/activate
pip install wheel
cd reptil; pip install .; cd ..
ninja
