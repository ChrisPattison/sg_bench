import pandas as pd
import numpy as np
import multiprocessing
import subprocess
import propanelib

def run_restart(schedule_file, instance_file): # schedule, instance
    restart_data = []
    try:
        command = ['propane_ptsvmc', '-m', 'pt', schedule_file, instance_file]
        output = subprocess.check_output(command, universal_newlines=True)
        output = output.split('\n')
        restart_data = propanelib.extract_data(output)
    except Exception as e:
        print(e)
    return restart_data

def get_data(schedule_file, instance_file, restarts=100, parallel=True):
    data = []

    pool = multiprocessing.Pool(None if parallel else 1)
    try:
        # data = pool.map_async(run_restart, [(schedule_file, instance_file) for i in range(restarts)])
        data = [pool.apply_async(run_restart, (schedule_file, instance_file)) for i in range(restarts)]
        data = [d.get() for d in data]
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
    else:
        pool.close()
        pool.join()

    data = pd.concat(data)
    return data
