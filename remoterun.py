import tempfile
import dispy
import pandas as pd
import copy
import bondfile
import propanelib
import pt_propanelib
import localrun

def run_instances(schedule, instances, dispyconf, restarts=400, statistics=True):
    cluster = dispy.JobCluster(localrun.run_restart, depends=[bondfile, propanelib, pt_propanelib], \
        loglevel=dispy.logger.CRITICAL, pulse_interval=2, reentrant=True, ping_interval=1,
        ext_ip_addr=dispyconf['ext_ip_addr'], nodes=dispyconf['nodes'])
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
        # print(i['results'])
        for index, df in enumerate(i['results']):
            df['restart'] = index
        i['results'] = pd.concat(i['results'])

        if not statistics:
            i['results'] = i['results'].groupby(['restart']).apply(lambda d: d[d['Total_Sweeps'] == d['Total_Sweeps'].max()]).reset_index(drop=True)
    return instances