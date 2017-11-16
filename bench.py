#!/usr/bin/python
import sys
import instance
import solve
import numpy as np

def bench(optimize_fields = True):
    if len(sys.argv) < 4:
        print('bench.py <instances> <T High> <T Low> <T Count>')
        quit()

    print('Loading instances from '+sys.argv[1])
    instances = instance.get_size(sys.argv[1])
    print('Solving...')
    tts = solve.bench_tempering(instances, (float(sys.argv[2]), float(sys.argv[3])), int(sys.argv[4]), optimize_fields = optimize_fields)
    print(tts[0])
    print(tts[1])
    print('Median TTS: '+str(np.median(tts[0])*tts[1])+' us')


if __name__ == "__main__":
    bench()