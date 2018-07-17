#!/usr/bin/env python3
from sg_bench.launch_restarts import runner_base, main
from psqa_bench import pa_propanelib

class runner(runner_base):
    solver = 'psqa_solve'
    def extract_data(self, output):
        return pa_propanelib.extract_data(output)

# Run the specified number of restarts on instance bond_file with schedule schedule_file
if __name__ == "__main__":
    main(runner)