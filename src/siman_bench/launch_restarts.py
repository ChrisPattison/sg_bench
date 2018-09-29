#!/usr/bin/env python3
from sg_bench.launch_restarts import get_pool
import argparse
import subprocess
import pandas as pd
import numpy as np
import json
import io
import re

# read schedule file to find info to pass to solver
# read output and expand

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


class runner:
    def run_restarts(self, schedule, bonds, restarts, ground_state_energy):
        with open(schedule) as config_file:
            config = json.load(config_file)
        pool = get_pool()

        restart_handles = [pool.apply_async(self._single_restart, 
            args=(config, schedule, bonds, k, ground_state_energy)) for k in range(restarts)]
        results = [job.get() for job in restart_handles]

        return pd.concat(results).reset_index()


    def _single_restart(self, config, schedule, bonds, index, ground_state_energy):
        sweeps = config['sweeps']
        beta = np.asscalar(np.max(config['param_set']['beta']['points']))
        command = [config['solver'], 
            '-l', bonds, 
            '-b0', str(np.asscalar(np.min(config['param_set']['beta']['points']))),
            '-b1', str(beta),
            '-s', str(sweeps),
            '-r', '1',
            '-v']
        output = subprocess.run(command, encoding='utf-8', stdout=subprocess.PIPE)
        run_time = float(re.search(r'(?m)^#work done in (?P<timing>(?:\d*)(?:\.\d*)?(?:e(?:\+|-)\d+)?) s$', output.stdout).group('timing'))
        output_table = '\n'.join(re.search(r'(?m)^(?!\#)(.*)$', output.stdout).groups())
        result = pd.read_csv(io.StringIO(output_table), delim_whitespace=True, header=None, names=['E_MIN', 'N_MIN', 'fraction', 'bondfile'])
        result['restart'] = index
        result['Total_Sweeps'] = sweeps
        result['Total_Walltime'] = run_time * 1e6
        result['Beta'] = beta
        return result

# Run the specified number of restarts on instance bond_file with schedule schedule_file
if __name__ == "__main__":
    main(runner)