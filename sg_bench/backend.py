import pandas as pd
import numpy as np
import os
import tempfile
import copy
import io
import subprocess
import multiprocessing
from sg_bench import bondfile, slurm, ssh


def get_backend(command, slurmconf = None):
    if slurmconf:
        return slurmrun(command, slurmconf)
    else:
        return localrun(command)

class localrun:
    def __init__(self, command):
        self._command = command

    def run_instances(self, schedule, instances, restarts, statistics=True):

        with tempfile.NamedTemporaryFile('w') as schedule_file:
            schedule_file.write(schedule)
            schedule_file.flush()
            for i in instances:
                # Write schedule
                with tempfile.NamedTemporaryFile('w') as bonds_file:
                    # Write bondfile
                    bondfile.write_bondfile(i['bonds'], bonds_file)
                    bonds_file.flush()

                    command = ('{} {} {} {} {}'
                        .format(self._command, schedule_file.name, bonds_file.name, restarts, ('' if statistics else '-p ' + str(i['target_energy']))))
                    output = subprocess.check_output(command, shell=True)
                    i['results'] = pd.read_csv(io.StringIO(output.decode('utf-8')))
                    if not statistics:
                        i['results'] = i['results'].groupby(['restart']).apply(lambda d: d[d['Total_Sweeps'] == d['Total_Sweeps'].max()]).reset_index(drop=True)

        return instances

    def close(self):
        return None


class slurmrun:
    def __init__(self, command, slurmconf):
        self._command = command
        self._ssh = ssh.ssh_wrapper(slurmconf)
        self._slurmconf = slurmconf

    def run_instances(self, schedule, instances, restarts, statistics=True):
        with slurm.slurm(self._slurmconf, ssh=self._ssh) as slurm_wrapper:
            commands = []

            schedule_path = slurm_wrapper.put_temp_file(schedule)
            for i in instances:
                bonds = io.StringIO()
                bondfile.write_bondfile(i['bonds'], bonds)
                instance_path = slurm_wrapper.put_temp_file(bonds.getvalue())
                output_path = instance_path + '.out'
                slurm_wrapper.reg_temp_file(output_path)
                commands.append('{} {} {} {} {} > {}'
                    .format(self._command, schedule_path, instance_path, restarts, ('' if statistics else '-p ' + str(i['target_energy'])), output_path))
                i['output_file'] = output_path

            slurm_wrapper.submit_job_array(commands)

            for i in instances:
                i['results'] = pd.read_csv(io.StringIO(self._ssh.get_string(i['output_file'])))
                if not statistics:
                    i['results'] = i['results'].groupby(['restart']).apply(lambda d: d[d['Total_Sweeps'] == d['Total_Sweeps'].max()]).reset_index(drop=True)

        return instances

    def close(self):
        self._ssh.close()