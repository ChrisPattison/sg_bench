#!/usr/bin/python
import numpy as np
import sys
import instance
import solve

if len(sys.argv) < 4:
    print('simple_bench.py <instances> <T High> <T Low> <T Count>')
    print('Field set uniformly distributed in 1/T if count is negative')
    quit()

print('Loading instances...')
instances = instance.get_size(sys.argv[1])
print('Solving...')
count = int(sys.argv[4])
if count < 0:
    temp_set = np.reciprocal(np.linspace(1./float(sys.argv[3]), 1./float(sys.argv[2]), -count))
else:
    temp_set = np.linspace(float(sys.argv[3]), float(sys.argv[2]), count)

tts = solve.get_opt_tts(instances, temp_set*instances[0]['bondscale'])
print(tts)
