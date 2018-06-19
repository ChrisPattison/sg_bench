#!/usr/bin/env python3
import numpy as np
import json
from quit_bench import pinput, solve

def bench(optimize_temp = True):
    config, instances, args = pinput.get_input('Benchmark solver')
    solver = solve.solve(config)

    tts = solver.bench_tempering(instances)
    if not config['machine_readable']:
        print(tts[0])
        print(tts[1])
        print('Median TTS: '+str(np.median(tts[0]))+' us')
    
    print(json.dumps(solver.get_full_data()))
if __name__ == "__main__":
    bench()