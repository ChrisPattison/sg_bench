#!/usr/bin/env python3
import bondfile
import pa_propanelib
import pandas as pd
import numpy as np
import warnings
import io
import sys
import tempfile
import multiprocessing
import io

# Creates and solves for the ground state of some spin glass instances given a template bondfile

def main():
    if len(sys.argv) < 3:
        print('curate.py <template bondfile> <count>')
        quit()

    template = bondfile.read_bondfile(sys.argv[1])
    count = int(sys.argv[2])

    np.random.seed()
    
    ground_states = []

    with tempfile.NamedTemporaryFile('w') as sch_file:
        # write schedule
        sch_file.write(pa_propanelib.make_schedule(20000, 2000))
        sch_file.flush()

	pool = multiprocessing.Pool(40)
	results = []
        for n in range(count):
            print(n)
            instance = template
            instance['J_ij'] = [np.random.normal() for i in instance['J_ij']]

            instance_file = sys.argv[1]+'.'+str(n)
            with io.open(instance_file, 'w') as bonds:
                bondfile.write_bondfile(instance, bonds)
            results.append(pool.apply_async(pa_propanelib.run_restart(sch_file.name, instance_file)))

        for n in range(count):
            result = results[n].get()
            if result[result['Beta']==result['Beta'].max()]['R_MIN'].max() > 100:
                ground_states.append({'instance':str(n), 'energy':float(result[result['Beta']==result['Beta'].max()]['E_MIN'])})
            else:
                warnings.warn('Did not find ground state for instance ' +str(n))
    ground_states = pd.DataFrame.from_records(ground_states)
    ground_states.to_csv(sys.argv[1]+'.energy', sep=' ')

if __name__ == '__main__':
    main()
