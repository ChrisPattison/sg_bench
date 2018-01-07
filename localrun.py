import pandas as pd
import numpy as np
import multiprocessing
import subprocess
import tempfile

def run_restart(schedule, instance, ground_energy = None): # schedule, instance
    import bondfile
    import pt_propanelib

    with tempfile.NamedTemporaryFile('w') as schedule_file:
        # Write schedule
        schedule_file.write(schedule)
        schedule_file.flush()
        with tempfile.NamedTemporaryFile('w') as bonds_file:
            # Write bondfile
            bondfile.write_bondfile(instance['bonds'], bonds_file)
            bonds_file.flush()

            # Run solver
            restart_data = []
            command = ['propane_ptsvmc', '-m', 'pt', schedule_file.name, bonds_file.name]
            # Optional ground state energy specification
            if ground_energy is not None:
                command.extend(['-p', str(ground_energy)])

            output = subprocess.check_output(command, universal_newlines=True)
            if 'returned non-zero exit status' in output:
                raise RuntimeException(output)
            output = output.split('\n')

            restart_data = pt_propanelib.extract_data(output)
    return restart_data

class localrun:
    def run_instances(schedule, instances, restarts=400, statistics=True):

        pool = multiprocessing.Pool()
        for i in instances:
            ground_energy = None if statistics else i['ground_energy']

            i['results'] = []
            for r in range(restarts):
                i['results'].append(pool.apply_async(run_restart (schedule, i, ground_energy)))
                print(i['results'])

        for i in instances:
            i['results'] = [job.get() for job in i['results']]
            # print(i['results'])
            for index, df in enumerate(i['results']):
                df['restart'] = index
            i['results'] = pd.concat(i['results'])

            if not statistics:
                i['results'] = i['results'].groupby(['restart']).apply(lambda d: d[d['Total_Sweeps'] == d['Total_Sweeps'].max()]).reset_index(drop=True)
        return instances