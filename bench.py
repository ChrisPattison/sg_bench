#!/usr/bin/python3
import solve
import pinput
import numpy as np

def bench(optimize_temp = True):
    config, instances = pinput.get_input('Benchmark solver')
    print('Solving...')
    tts = solve.bench_tempering(instances,\
        beta = (config['beta']['min'], config['beta']['max']), \
        temp_count = config['beta']['count'], \
        field_strength = config.get('field_strength', 1.0), \
        profile = config['profile'], \
        restarts = config.get('restarts', 100), \
        optimize_temp = optimize_temp)
    print(tts[0])
    print(tts[1])
    print('Median TTS: '+str(np.median(tts[0])*tts[1])+' us')


if __name__ == "__main__":
    bench()