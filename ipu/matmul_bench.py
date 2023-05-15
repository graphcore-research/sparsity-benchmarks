# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import argparse
import errno
import json
import os
import sys
import shlex
import subprocess
import fnmatch as fnm
from reptil import Reptil, open_report


def make_arg_parser(parser=None):
    if not parser:
        parser = argparse.ArgumentParser(
            prog="matmul",
            description="Argument parser for matrix multiplication utility",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    parser.add_argument(
        "--implementation",
        type=str,
        default="dense",
        choices=["dense", "static-block-sparse", "dynamic-block-sparse"],
        help="Which implementation to run."
    )

    parser.add_argument(
        "-m",
        "--output",
        type=int,
        required=True,
        help="The number of rows for the left hand side matrix"
    )

    parser.add_argument(
        "-k",
        "--input",
        type=int,
        required=True,
        help="The number of rows for the right hand side matrix and \
              the number of columns for the left hand side matrix"
    )

    parser.add_argument(
        "-n",
        "--batch",
        type=int,
        required=True,
        help="The number of columns for the right hand side matrix"
    )

    parser.add_argument(
        "--type",
        type=str,
        default="half",
        choices=["half", "float"],
        help="Type of the floating point representation (half|float)"
    )

    parser.add_argument(
        "--partials",
        required=False,
        help="Switch to half partial accumulators (half)"
    )

    parser.add_argument(
        "--device-type",
        type=str,
        default="Hw",
        choices=["Hw", "IpuModel", "IpuModel2", "IpuModel21", "IpuModelConfig"],
        help="Which device type should be used."
    )

    parser.add_argument(
        "--available-memory-proportion",
        type=float,
        default=0.6,
        required=False,
        help="The available memory proportion to be used when executing"
    )

    parser.add_argument(
        "--density",
        type=float,
        default=1.0,
        required=False,
        help="Proportion of non-zero element for sparse matrices"
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=1,
        choices=[1, 4, 8 , 16],
        required=False,
        help="Block size for sparse operations"
    )

    parser.add_argument(
        "--profile-dir",
        type=str,
        default="./profile",
        help="The folder where to store the profile file"
    )

    parser.add_argument(
        "--model-file",
        type=str,
        help="The path to the JSON model config file"
    )

    return parser


def output_computeset_tile_cycles(rt: Reptil, flopc, freq, op_name):
    op_cycles = 0
    total_cycles = 0
    no_active_tiles = 0
    overl_compute_cycles = 0
    overl_exchange_cycles = 0

    try:
        cycles_dict = rt.cycles.computeset_tile()
        cycles_matmul_int = rt.cycles.intervals([op_name])
    except IndexError:
        # If an OOM error was thrown then there won't be any execution runs
        # to extract cycle metrics from
        return

    for t in cycles_matmul_int.get("compute", []):
        overl_compute_cycles += int(t[1]) - int(t[0])
    for t in cycles_matmul_int.get("exchange", []):
        overl_exchange_cycles += int(t[1]) - int(t[0])

    for k in cycles_dict.keys():
        per_cs = sum(cycles_dict[k])
        total_cycles += per_cs
        if fnm.fnmatch(k, op_name):
            op_cycles += per_cs
            at_cs = len([x for x in cycles_dict[k] if x != 0])
            if at_cs > no_active_tiles:
                no_active_tiles = at_cs

    print(f"Total compute cycles: {int(total_cycles)}")
    print(f"Total matmul compute cycles: {int(op_cycles)}")
    print(f"Overlapped matmul compute cycles: {overl_compute_cycles}")
    print(
        f"Overlapped matmul compute and exchange cycles: "
        f"{overl_compute_cycles + overl_exchange_cycles}")
    print(f"Number of active tiles: {int(no_active_tiles)}")
    if op_cycles > 0:
        print(f"FLOPs per matmul compute cycle: {flopc/op_cycles}")
    if overl_compute_cycles > 0:
        print(f"Compute TFLOPS: {flopc/overl_compute_cycles*freq/1000}")
    if overl_compute_cycles + overl_exchange_cycles > 0:
        print(
            f"Compute and exchange TFLOPS: "
            f"{flopc/(overl_compute_cycles + overl_exchange_cycles)*freq/1000}"
            )


def output_program_cycles(rt: Reptil):
    cycles_dict = rt.cycles.program()
    execute_cycles = 0
    for k in cycles_dict.keys():
        if "Execute" in k:
            execute_cycles += cycles_dict[k]
    print(f"Total execute cycles: {int(execute_cycles)}")


def output_program_memory(rt: Reptil):
    print(f"Total memory: {int(sum(rt.memory.total.ipus))}")
    print(
        f"Total always live memory per IPU: "
        f"{int(sum(rt.memory.always_live.ipus))}")
    print(
        f"Max not always live memory per tile: "
        f"{int(max(max(rt.memory.not_always_live.tiles)))}")
    print(f"Peak liveness per IPU: {int(rt.memory.peak_liveness)}")
    print(f"Total exchange memory: {int(sum(rt.memory.exchange.ipus))}")
    print(f"Total vertex memory: {int(sum(rt.memory.vertex.ipus))}")


if __name__ == '__main__':

    parser = make_arg_parser()
    args = parser.parse_args()
    (m, n, k) = (args.output, args.batch, args.input)

    cmd = f"./matmul_bench --implementation {args.implementation}"
    cmd += f" -m {m} -k {k} -n {n} --data-type {args.type}"
    cmd += f" --density {args.density}"
    cmd += f" --block-size {args.block_size}"
    cmd += (f" --available-memory-proportion"
            f" {args.available_memory_proportion}")
    if args.partials == "half" and args.type == "half":
        cmd += " --partials-type half"
    cmd += f" --profile-dir {args.profile_dir}"
    cmd += f" --device-type {args.device_type}"
    if args.model_file is not None:
        cmd += f" --model-file {args.model_file}"

    print(cmd)
    args_list = shlex.split(cmd)

    proc = subprocess.Popen(args_list)
    proc.wait()

    # Output a single flag to indicate whether the run was successful.
    print(f"Success: {proc.returncode == 0}")
    if proc.returncode != 0:
        exit(1)

    # Profile reports are generated and can contain useful information,
    # even for unsuccessful runs (e.g. OOM errors or validation failures),
    # so if there's one available then process it.
    profile_report_path = os.path.join(args.profile_dir, 'profile.pop')
    if not os.path.exists(profile_report_path):
        print(f"Profile file {profile_report_path} not found")
        exit(1)

    rt = open_report(profile_report_path)
    output_computeset_tile_cycles(rt, 2*m*n*k*args.density, rt.report.compilation.target.clockFrequency/1000000000, "*matmul*")
    output_program_memory(rt)
    print(f"Clock: {rt.report.compilation.target.clockFrequency} Hz")
