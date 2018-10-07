from siman_bench import solve, launch_restarts
from pathlib import Path
from collections import namedtuple
import json

def test_siman_bench(smoketest_fixture, config_fixture):
    # Load test specific configuration
    with open(Path(config_fixture['config_path']).parents[0]/'siman_config.json') as config_file:
        config = json.load(config_file)
    # Merge with smoketest config overriding keys as necessary
    test_config = {**config_fixture['config'], **config}
    solver = solve.solve(test_config)
    output = smoketest_fixture.run(solver)
    assert(True)

def test_siman_launch_restarts():
    arg_tuple = namedtuple('arg_tuple', ['schedule_file', 'bond_file', 'restarts', 'ground_state_energy'])
    launch_restarts.main(launch_restarts.runner, arg_tuple(
        schedule_file='test_data/launch_restarts/siman_launch_restarts.json',
        bond_file='test_data/launch_restarts/sk_wishart_48_0',
        restarts=100,
        ground_state_energy=float('NaN')
    ))
