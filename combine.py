#!/usr/bin/python3
import json
import numpy as np
import pathlib
import sys
import warnings

def combine():
    conf_list = pathlib.Path(sys.argv[1])
    out_files = []
    with conf_list.open('r') as f:
        out_files = [pathlib.Path(s.strip()+'.out') for s in f.readlines()]
    
    combined = None
    for output in out_files:
        with output.open('r') as f:
            output = json.load(f)
        if combined is None:
            combined = output
        
        # These should be exactly equal
        if not np.array_equal(combined['field_set'], output['field_set']):
            warnings.warn('Field set is inconsistent')

        if combined['beta'] is not output['beta']:
            warnings.warn('Beta is inconsistent')

        combined['tts'] += combined['tts']
        combined['time_per_sweep'] += combined['time_per_sweep']

    combined['time_per_sweep'] /= len(out_files)

    print(json.dumps(combined))

if __name__ == "__main__":
    combine()