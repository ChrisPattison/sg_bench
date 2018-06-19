#!/usr/bin/env python3
import numpy as np
import pandas as pd
import subprocess
import multiprocessing
import argparse
import io
from quit_bench import pt_propanelib

def single_restart(schedule, bonds, index, ground_state_energy):
    command = ['quit_solve', schedule, bonds]
    if ground_state_energy:
        command.extend(['-p', str(ground_state_energy)])
    output = subprocess.check_output(command, universal_newlines=True)
    result = pt_propanelib.extract_data(output.split('\n'))
    result['restart'] = index
    return result

def run_restarts(schedule, bonds, restarts, ground_state_energy):
    cores = -1
    # Attempt to find out the number of physical cores using psutil
    try:
        import psutil
        cores = psutil.cpu_count(logical=False)
    except ImportError:
        cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cores)

    restart_handles = [pool.apply_async(single_restart, args=(schedule, bonds, k, ground_state_energy)) for k in range(restarts)]
    results = [job.get() for job in restart_handles]

    return pd.concat(results).reset_index()

# Run the specified number of restarts on instance bond_file with schedule schedule_file
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("schedule_file", help="Name of the schedule file")
    parser.add_argument("bond_file", help="Name of the bonds file")
    parser.add_argument("restarts", help="Total number of restarts", type=int)
    parser.add_argument("-p", help="Ground state energy to pass to solver", type=float, dest='ground_state_energy')

    args = parser.parse_args()
    results = run_restarts(args.schedule_file, args.bond_file, args.restarts, args.ground_state_energy)
    output = io.StringIO()
    results.to_csv(output)
    print(output.getvalue())
