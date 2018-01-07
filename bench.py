#!/usr/bin/python3
import sys
import instance
import solve
import numpy as np
import json
import io
import pathlib

def bench(optimize_temp = True):
    if np.any(['-h' in s for s in sys.argv]) or len(sys.argv) < 2:
        help = \
'''Copyright (c) 2017 C. Pattison

Configuration is the path to a JSON file that includes the following keys:

instances : string
beta :
    count : int
    min : float
    max : float
field_strength : float
profile : float array

bench.py <configuration>
'''
        print(help)
        quit()


    config_path = pathlib.Path(sys.argv[1])
    config = {}
    with io.open(str(config_path.resolve()), 'r') as config_file:
        config = json.load(config_file)

    instance_path = config_path.parents[0] / config['instances']
    print('Loading instances from '+str(instance_path.resolve()))
    instances = instance.get_instance_set(str(instance_path.resolve()))
    print('Solving...')
    tts = solve.bench_tempering(instances,\
        beta = (config['beta']['min'], config['beta']['max']), \
        temp_count = config['beta']['count'], \
        field_strength = config['field_strength'], \
        profile = config['profile'], \
        restarts = config.get('restarts', 100), \
        optimize_temp = optimize_temp)
    print(tts[0])
    print(tts[1])
    print('Median TTS: '+str(np.median(tts[0])*tts[1])+' us')


if __name__ == "__main__":
    bench()