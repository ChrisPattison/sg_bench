import sys
import json
import io
import pathlib
import numpy as np
from sg_bench import instance

def get_input(help_text, fetch_instances=True):
    args = sys.argv
    if '-h' in args or len(sys.argv) < 2:
        help = 'Copyright (c) 2017 C. Pattison\n' + help_text + \
'''
Configuration is the path to a JSON file that includes the following keys:

instances : string
beta :
    count : int
    min : float
    max : float
field_strength : float
profile : float array

<util>.py <configuration> <optional args>
'''
        print(help)
        quit()

    machine_readable = '-m' in args

    # Strip out flags
    args = [a for a in args if not a.startswith('-')]
    flags = [a for a in args if a.startswith('-')]
    config_path = pathlib.Path(args[1])
    config = {}
    with io.open(str(config_path.resolve()), 'r') as config_file:
        config = json.load(config_file)
    
    # Optional template config
    # Entries in config overload entries in the template config
    config_template_path = config.get('template_config', None)
    if config_template_path:
        with (config_path.resolve().parent / pathlib.Path(config_template_path)).open() as template_file:
            template_config = json.load(template_file)
        config = {**template_config, **config}
    # Load instances
    instance_path = pathlib.Path(config['instances'])
    if not instance_path.is_absolute():
        instance_path = config_path.parents[0] / instance_path

    if fetch_instances:
        if not machine_readable:
            print('Loading instances from '+str(instance_path.resolve()))
        instances = instance.get_instance_set(str(instance_path.resolve()))
    else:
        instances = instance.read_instance_list(str(instance_path.resolve()))
    
    config['machine_readable'] = machine_readable

    return config, instances, args, flags
