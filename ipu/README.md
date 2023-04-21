### A simple application for running matmuls and extracting execution metrics from profile files

#### Requirements:
  * Poplar SDK downloaded and activated
  * clang for compiling
  * ninja for building
  * python >= 3.8 for running
  * reptil (supplied)

#### Build instructions:  
  `source install.sh` in the current folder to:  
  1. create a virtual environment and activate it
  2. install reptil
  3. build matmul_bench app

#### Running:  
  The executable can be run by itself:  
    `./matmul_bench --help`  
  or through the python wrapper:  
    `python matmul_bench.py --help`

  The python wrapper is necessary for extracting the execution metrics.
