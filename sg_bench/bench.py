import numpy as np
import json
from sg_bench import pinput

def bench(solve, optimize_temp = True):
    config, instances, args, _ = pinput.get_input('Benchmark solver')
    solver = solve(config)
    tts = solver.bench(instances)
    if not config['machine_readable']:
        print(tts[0])
        print(tts[1])
        print('Median TTS: '+str(np.median(tts[0]))+' us')
    
    print(json.dumps(solver.get_full_data()))
