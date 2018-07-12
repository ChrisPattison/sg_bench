import numpy as np
import pandas as pd
import json
from quit_solve import pinput

def observe(solve, optimize_temp = True):
    config, instances, args, _ = pinput.get_input('Compute observables for instance class')
    solver = solve(config)
    run_data = solver.observe(instances)
    if not config['machine_readable']:
        print(run_data)
    else:
        full_data = solver.get_full_data()
        full_data['data'] = run_data.to_dict(orient='list')
        print(json.dumps(full_data))
    
if __name__ == "__main__":
    observe()
