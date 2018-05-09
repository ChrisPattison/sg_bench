#!/usr/bin/env python3
import numpy as np
import pandas as pd
import subprocess
import multiprocessing
import argparse
import io
import pa_propanelib

def single_restart(schedule, bonds, index):
    command = ['quantum_propane', schedule, bonds]
    output = subprocess.check_output(command, universal_newlines=True)
    result = pa_propanelib.extract_data(output.split('\n'))
    result['restart'] = index
    return result

def run_restarts(schedule, bonds, restarts):
    cores = -1
    # Attempt to find out the number of physical cores using psutil
    try:
        import psutil
        cores = psutil.cpu_count(logical=False)
    except ImportError:
        cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cores)

    restart_handles = [pool.apply_async(single_restart, args=(schedule, bonds, k)) for k in range(restarts)]
    results = [job.get() for job in restart_handles]

    return pd.concat(results).reset_index()

# Run the specified number of restarts on instance bond_file with schedule schedule_file
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("schedule_file", help="Name of the schedule file")
    parser.add_argument("bond_file", help="Name of the bonds file")
    parser.add_argument("restarts", help="Total number of restarts", type=int)

    args = parser.parse_args()
    results = run_restarts(args.schedule_file, args.bond_file, args.restarts)
    output = io.StringIO()
    results.to_csv(output)
    print(output.getvalue())
