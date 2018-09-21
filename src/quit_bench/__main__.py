#!/usr/bin/env python3
from quit_bench import solve
from sg_bench.bench import bench

def main():
    bench(solve.solve)

if __name__ == '__main__':
    main()