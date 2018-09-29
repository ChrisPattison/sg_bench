#!/usr/bin/env python3
import numpy as np
import pandas as pd
import subprocess
import multiprocessing
import argparse
import io
import sys

def get_pool():
    cores = -1
    # Attempt to find out the number of physical cores using psutil
    try:
        import psutil
        cores = psutil.cpu_count(logical=False)
    except ImportError:
        cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cores)
    return pool

class runner_base:
    def _single_restart(self, solver, schedule, bonds, index, ground_state_energy):
        command = [solver, schedule, bonds]
        if ground_state_energy:
            command.extend(['-p', str(ground_state_energy)])
        output = subprocess.run(command, encoding='utf-8', stdout=subprocess.PIPE)
        result = self.extract_data(output.stdout.split('\n'))
        result['restart'] = index
        return result

    def run_restarts(self, schedule, bonds, restarts, ground_state_energy):
        pool = get_pool()

        restart_handles = [pool.apply_async(self._single_restart, 
            args=(self.solver, schedule, bonds, k, ground_state_energy)) for k in range(restarts)]
        results = [job.get() for job in restart_handles]

        return pd.concat(results).reset_index()

# Run the specified number of restarts on instance bond_file with schedule schedule_file
def main(runner):

    parser = argparse.ArgumentParser()

    parser.add_argument("schedule_file", help="Name of the schedule file")
    parser.add_argument("bond_file", help="Name of the bonds file")
    parser.add_argument("restarts", help="Total number of restarts", type=int)
    parser.add_argument("-p", help="Ground state energy to pass to solver", type=float, dest='ground_state_energy')

    args = parser.parse_args()
    results = runner().run_restarts(args.schedule_file, args.bond_file, args.restarts, args.ground_state_energy)
    output = io.StringIO()
    results.to_csv(output)
    print(output.getvalue())
