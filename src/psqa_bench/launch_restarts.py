#!/usr/bin/env python3
from sg_bench.launch_restarts import runner_base, main
from psqa_bench import pa_propanelib
import subprocess

class runner(runner_base):
    def _single_restart(self, solver, schedule, bonds, index, ground_state_energy):
        # psqa_bench does not support the -p flag since it's sequential
        command = [solver, schedule, bonds]
        output = subprocess.run(command, encoding='utf-8', stdout=subprocess.PIPE)
        result = self.extract_data(output.stdout.split('\n'))
        result['restart'] = index
        return result

    solver = 'psqa_solve'
    def extract_data(self, output):
        return pa_propanelib.extract_data(output)

# Run the specified number of restarts on instance bond_file with schedule schedule_file
if __name__ == "__main__":
    main(runner)