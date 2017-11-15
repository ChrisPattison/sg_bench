import tempfile
import dispy
import pandas as pd
import copy
import bondfile
import propanelib
import pt_propanelib

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
            try:
                output = subprocess.check_output(command, universal_newlines=True)
                output = output.split('\n')
                restart_data = pt_propanelib.extract_data(output)
            # Catch misc errors
            except subprocess.CalledProcessError as e:
                print(e)
                print(e.output)
            except Exception as e:
                print(e)
    return restart_data

def run_instances(schedule, instances, restarts=400, statistics=True):
    cluster = dispy.JobCluster(run_restart, nodes=['tempeh.tamu.edu'], depends=[bondfile, propanelib, pt_propanelib], loglevel=dispy.logger.CRITICAL)
    for i in instances:
        ground_energy = None if statistics else i['ground_energy']

        # deepcopy required since i is not picklable
        i_copy = copy.deepcopy(i)
        i['results'] = []
        for r in range(restarts):
            i['results'].append(cluster.submit(schedule, i_copy, ground_energy = ground_energy))

    cluster.wait()
    for i in instances:
        for job in i['results']:
            if job.exception is not None:
                print(job.exception)
        i['results'] = [job() for job in i['results']]
        for index, df in enumerate(i['results']):
            df['restart'] = index
        i['results'] = pd.concat(i['results'])

        if not statistics:
            i['results'] = i['results'].groupby(['restart']).apply(lambda d: d[d['Total_Sweeps'] == d['Total_Sweeps'].max()]).reset_index(drop=True)
    return instances