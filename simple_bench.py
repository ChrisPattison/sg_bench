#!/usr/bin/python
import numpy as np
import sys
import instance
import solve

if sys.argc < 4:
    print('simple_bench.py <instances> <T High> <T Low> <T Count>')
    print('Field set uniformly distributed in 1/T if count is negative')
    quit()

print('Loading instances...')
instances = instance.get_size(sys.argv[1])
print('Solving...')
count = int(sys.argv[4])
if count < 0:
    count = -count
    temp_set = np.reciprocal(np.linspace(1./float(sys.argv[2]), 1./float(sys.argv[1]), -count))
else:
    temp_set = np.linspace(float(sys.argv[2]), float(sys.argv[1]), count)

tts = solve.get_opt_tts(instances, temp_set)
print(tts)