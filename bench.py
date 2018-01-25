#!/usr/bin/python3
import solve
import pinput
import numpy as np
import json

def bench(optimize_temp = True):
    config, instances, args = pinput.get_input('Benchmark solver')
    solver = solve.solve(config)

    tts = solver.bench_tempering(instances)
    if not config['machine_readable']:
        print(tts[0])
        print(tts[1])
        print('Median TTS: '+str(np.median(tts[0])*tts[1])+' us')
    
    print(json.dumps(solver.get_full_data()))
if __name__ == "__main__":
    bench()