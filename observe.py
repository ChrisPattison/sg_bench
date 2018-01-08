#!/usr/bin/python3
import solve
import pinput
import numpy as np
import pandas as pd

def observe(optimize_temp = True):
    config, instances = pinput.get_input('Compute observables for instance class')
    solver = solve.solve(config)
    run_data = solver.observe(instances)
    print(run_data)
    
if __name__ == "__main__":
    observe()