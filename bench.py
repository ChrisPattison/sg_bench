#!/usr/bin/python
import sys
import instance
import solve

print('Loading instances...')
instances = instance.get_size(sys.argv[1])
print('Solving...')
tts = solve.bench_tempering(instances)
print(tts)