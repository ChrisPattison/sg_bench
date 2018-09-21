from sg_bench import instance, backend
from pathlib import Path
import pytest
import json

class smoketest:
    def __init__(self, instance_path, config_path):
        instance_path = Path(instance_path)
        config_path = Path(config_path)
        if not instance_path.is_absolute():
            instance_path = config_path.parents[0] / instance_path        

        self._instances = instance.get_instance_set(str(instance_path.resolve()))

    def run(self, solver):
        tts = solver.bench(self._instances)
        return tts

@pytest.fixture(scope='module', params=['test_data'])
def config_fixture(request):
    config_path = Path(request.param) / Path('config.json')
    with open(config_path) as test_config_file:
        test_config = json.load(test_config_file)
    test_config['machine_readable'] = True
    return {'config':test_config, 'config_path':config_path}

@pytest.fixture(scope='module')
def smoketest_fixture(config_fixture):
    return smoketest(config_fixture['config']['instances'], config_fixture['config_path'])

@pytest.fixture(scope='module', params=['slurm', 'local'])
def backend_fixture(config_fixture, request):
    return backend.get_backend(slurmconf=(config_fixture['config']['slurm'] if request.param == 'slurm' else None))