#!/usr/bin/python
import sys
import instance
import solve
import numpy as np

if len(sys.argv) < 4:
    print('simple_bench.py <instances> <T High> <T Low> <T Count>')
    quit()

print('Loading instances...')
instances = instance.get_size(sys.argv[1])
print('Solving...')
tts = solve.bench_tempering(instances, (float(sys.argv[2]), float(sys.argv[3])), int(sys.argv[4]))
print(tts)
print(np.median(tts))
