import pandas as pd
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

def make_schedule(sweeps, steps, bondscale, bins=None):
    beta = 0.51/bondscale
    mc_sweeps = 1

    schedule = {'sweeps':int(sweeps), 'solver_mode':True, 'uniform_init':True, \
        'schedule':[{ 'beta':beta, 'gamma':s, 'metropolis':1, 'microcanonical':mc_sweeps } for s in steps],\
        'bin_set':([int(sweeps)] if bins is None else bins)}
    return json.dumps(schedule, indent=1)
