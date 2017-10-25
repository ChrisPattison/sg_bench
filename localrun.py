import pandas as pd
import numpy as np
import multiprocessing
import subprocess
import propanelib

def run_restart(schedule_file, instance_file, ground_energy = None): # schedule, instance
    restart_data = []
    command = ['propane_ptsvmc', '-m', 'pt', schedule_file, instance_file]
    if ground_energy is not None:
        command.extend(['-p', str(ground_energy)])
    try:
        output = subprocess.check_output(command, universal_newlines=True)
        output = output.split('\n')
        restart_data = propanelib.extract_data(output)
    except subprocess.CalledProcessError as e:
        print(e)
        print(e.output)
    except Exception as e:
        print(e)
    return restart_data

def get_data(schedule_file, instance_file, restarts=100, parallel=True, ground_energy=None):
    data = []

    pool = multiprocessing.Pool(None if parallel else 1)
    try:
        # data = pool.map_async(run_restart, [(schedule_file, instance_file) for i in range(restarts)])
        data = [pool.apply_async(run_restart, (schedule_file, instance_file, ground_energy)) for i in range(restarts)]
        data = [d.get() for d in data]
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
    else:
        pool.close()
        pool.join()

    for i in range(len(data)):
        data[i]['restart'] = i
    data = pd.concat(data)
    return data
