#!/usr/bin/env python3
from psqa_bench import solve
from sg_bench.observe import observe

if __name__ == '__main__':
    observe(solve.solve)