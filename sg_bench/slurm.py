import time
import io
import pandas as pd
from sg_bench import ssh

# Wrapper to manage SLURM job arrays
# Also keeps track of temp files and deletes them
class slurm:
    def __init__(self, slurmconf, ssh = None):
        self._wait_period = slurmconf.get('wait_period', 10) # in seconds
        self._remote_work_dir = slurmconf['work_dir'] + '/'
        self._temp_file_list = []
        self._ssh_borrow = bool(ssh)
        self._ssh = ssh.ssh_wrapper(slurmconf) if not ssh else ssh
        with open(slurmconf['sub_script'], 'r') as sub_script:
            self._submission_script = sub_script.read()


    # Check the status of the job array
    # Returns true if the job array is still running
    def _get_job_array_status(self, job_id):
        _, stdout, _ = self._ssh.exec_command('squeue -j {} -r'.format(job_id))
        job_list = pd.read_csv(io.StringIO(stdout.read().decode('utf-8')), delim_whitespace=True)
        return len(job_list['JOBID']) != 0

    def _make_sub_script(self, commands):
        contents = self._submission_script
        contents += '\n' + commands
        return self.put_temp_file(contents)

    def _delete_temp_files(self):
        for temp_file in self._temp_file_list:
            self._ssh.delete_file(temp_file)
        self._temp_file_list = []

    def put_temp_file(self, contents):
        filename = self._ssh.put_string(contents, remote_dir=self._remote_work_dir)
        self.reg_temp_file(filename)
        return filename
        
    def reg_temp_file(self, path):
        self._temp_file_list.append(path)

    # Submit job array and wait for completion
    def submit_job_array(self, task_list):
        command_file = self.put_temp_file('\n'.join(task_list))
        sub_script = self._make_sub_script('eval $(sed "${{SLURM_ARRAY_TASK_ID}}q;d" "{}")'.format(command_file))

        _, stdout, _ = self._ssh.exec_command('sbatch --parsable --array=1-{} --chdir="{}" "{}"'.format(len(task_list), self._remote_work_dir, sub_script))
        job_id = int(stdout.read().decode('utf-8'))

        while self._get_job_array_status(job_id):
            time.sleep(self._wait_period)

    def get_work_dir(self):
        return self._remote_work_dir

    def close(self):
        self._ssh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if not self._ssh_borrow:
            self._ssh.close()
        self._delete_temp_files()
