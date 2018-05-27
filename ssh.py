import paramiko
import getpass
import tempfile
import time
import random
import string
import numpy as np

class ssh_wrapper:
    def __init__(self, config):
        self._hostname = config['hostname']
        self._user = config['user']

        self._timeout = config.get('timeout', 10)
        self._check_duration = config.get('timeout_check', 10)

        self._client= paramiko.SSHClient()
        self._password = getpass.getpass(self._user+'@'+self._hostname+': ')

        self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self._connect_to_host()
        self._open_sftp_session()

        self._last_check = time.time()

    def _connect_to_host(self):
        self._client.connect(
            self._hostname, 
            username=self._user, 
            password=self._password, 
            allow_agent = False, 
            timeout=self._timeout)

    def _open_sftp_session(self):
        self._sftp_client = self._client.open_sftp()
    
    def _connection_active(self):
        if (time.time() - self._last_check) < self._check_duration:
            return True
        else:
            try:
                _, stdout, _ = self._client.exec_command('echo')
                stdout.read()
                return True
            except SSHException:
                return False

    def _make_active(self):
        status = self._connection_active()
        if not status:
            self._connect_to_host()
        return status

    def _make_sftp_active(self):
        status = self._make_active()
        if not status:
            self._open_sftp_session()
        return status

    # Generate a unique filename on the remote host
    # Race condition exists but is unlikely (95 bits of entropy)
    def _get_temp_file_name(self, remote_dir):
        remote_filelist = self._sftp_client.listdir(remote_dir)
        filename = None
        while True:
            filename = 'tmp.'+''.join(np.random.choice(list(string.ascii_letters + string.digits), size=16))
            if not np.any([f.endswith(filename) for f in remote_filelist]):
                break
        return remote_dir + '/' + filename

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def exec_command(self, command):
        self._make_active()
        return self._client.exec_command(command)

    # Write the string to a file on the remote host and return the path
    # Uses _get_temp_file_name to generate a filename on the remote host in remote_dir if remote_filename is not specified.
    # remote_dir is ignored if remote_filename is specified
    def put_string(self, contents, remote_filename = None, remote_dir = None):
        self._make_sftp_active()
        with tempfile.NamedTemporaryFile('w') as local_file:
            local_file.write(contents)
            local_file.flush()
            if not remote_filename:
                remote_filename = self._get_temp_file_name( remote_dir)
            self.put_file(local_file.name, remote_filename)
        return remote_filename

    # Get contents of file on remote host
    def get_string(self, remote_filename):
        contents = None
        with tempfile.NamedTemporaryFile('r') as local_file:
            self.get_file(remote_filename, local_file.name)
            local_file.seek(0)
            contents = local_file.read()
        return contents

    # Transfer file to remote host
    def put_file(self, local_filename, remote_filename):
        self._make_sftp_active()
        self._sftp_client.put(local_filename, remote_filename)
        return remote_filename

    # Transfer file from remote host
    def get_file(self, remote_filename, local_filename):
        self._make_sftp_active()
        self._sftp_client.get(remote_filename, local_filename)

    # Delete file on remote host
    def delete_file(self, remote_filename):
        self._make_active()
        self._client.exec_command('rm -f ' + remote_filename)

    # Close the SSHClient
    def close(self):
        self._client.close()