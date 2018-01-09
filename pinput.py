import sys
import instance
import json
import io
import pathlib
import numpy as np

def get_input(help_text):
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

<util>.py <configuration>
'''
        print(help)
        quit()

    machine_readable = '-m' in args

    # Strip out flags
    args = [a for a in args if not a.startswith('-')]
    config_path = pathlib.Path(args[1])
    config = {}
    with io.open(str(config_path.resolve()), 'r') as config_file:
        config = json.load(config_file)
    
    instance_path = config_path.parents[0] / config['instances']
    if not machine_readable:
        print('Loading instances from '+str(instance_path.resolve()))
    instances = instance.get_instance_set(str(instance_path.resolve()))
    
    config['machine_readable'] = machine_readable
    return config, instances