from pt_bench import solve
from pathlib import Path
import json

def test_pt_bench(smoketest_fixture, config_fixture):
    # Load test specific configuration
    with open(Path(config_fixture['config_path']).parents[0]/'pt_config.json') as config_file:
        config = json.load(config_file)
    # Merge with smoketest config overriding keys as necessary
    test_config = {**config_fixture['config'], **config}
    solver = solve.solve(test_config)
    output = smoketest_fixture.run(solver)
    assert(True)