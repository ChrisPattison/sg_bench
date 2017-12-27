import pandas as pd
import numpy as np
import json
import io

magic = '%%%---%%%'
comment = '#'
histstart = '|'
config_parse_error = 'Config parsing failed.'

def extract_data(output):
    if config_parse_error in ''.join(output):
        raise RuntimeError('Solver failed to parse input config')
    split = 0
    # find magic string
    for num, l in enumerate(output, 0):
        split = num
        if magic in l:
            break
    # get lines to write back later
    lines = output
    
    # remove histograms
    lines = lines[0:split]
    # strip out whitespacee and comments
    lines = [l.split(comment)[0].strip() for l in lines]
    # remove empty lines
    lines = [l for l in lines if len(l)]
    # reassemble in buffer
    buff = io.StringIO(unicode('\n'.join(lines)))
    buff.seek(0)
    data = pd.read_csv(buff, delim_whitespace=True)
    return data