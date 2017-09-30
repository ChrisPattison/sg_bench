import pandas as pd
import numpy as np
import tempfile
import bondfile
import multiprocessing
import subprocess
import json
import io

magic = '%%%---%%%'
comment = '#'
histstart = '|'

def extract_data(output):
    split = 0
    # find magic string
    for num, l in enumerate(output, 0):
        split = num
        if magic in l:
            break
    # get lines to write back later
    lines = output
    
    # remove histograms
    lines = lines[0:split]
    # strip out whitespacee and comments
    lines = [l.split(comment)[0].strip() for l in lines]
    # remove empty lines
    lines = [l for l in lines if len(l)]
    # reassemble in buffer
    buff = io.StringIO(unicode('\n'.join(lines)))
    buff.seek(0)
    data = pd.read_csv(buff, delim_whitespace=True)
    return data

def run_restart(fileset): # schedule, instance
    command = ['propane_ptsvmc', '-m', 'pt', fileset[0], fileset[1]]
    output = subprocess.check_output(command, universal_newlines=True)
    output = output.split('\n')
    restart_data = extract_data(output)
    return restart_data

def get_data(schedule_file, instance_file, restarts=100, parallel=True):
    data = []

    pool = multiprocessing.Pool(None if parallel else 1)
    data = pool.map(run_restart, [(schedule_file, instance_file) for i in range(restarts)])

    data = pd.concat(data)
    return data

def make_schedule(sweeps, steps, bondscale):
    beta = 0.51/bondscale
    mc_sweeps = 1

    schedule = {'sweeps':int(sweeps), 'solver_mode':True, 'uniform_init':True, 'schedule':[{ 'beta':beta, 'gamma':s, 'metropolis':1, 'microcanonical':mc_sweeps } for s in steps]}
    return json.dumps(schedule, indent=1)

def get_tts(instances):
    results = []

    # make schedule
    schedule = make_schedule(256, np.linspace(18, 0, 32), instances[0]['bondscale'])
    with tempfile.NamedTemporaryFile('w') as sch_file:
        # write bondfile
        sch_file.write(schedule)
        sch_file.flush()

        for i in instances:
            with tempfile.NamedTemporaryFile('w') as bonds_file:
                # write bond file
                bondfile.write_bondfile(i['bonds'], bonds_file)
                bonds_file.flush()
                
                # run restarts
                i['results'] = get_data(sch_file.name, bonds_file.name)

    tts = []
    for i in instances:
        success_prob = np.mean(np.isclose(i['ground_energy'], i['results']['E_MIN']))
        tts.append(np.mean(i['results']['Total_Sweeps'])*np.log(1-.99)/np.log(1. - success_prob))
    
    return tts