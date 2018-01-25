#!/usr/bin/python3
import pinput
import instance
import pathlib
import numpy as np
import json

def partition():
    config, instance_list, args = pinput.get_input('Partition instances\n partition.py <config> <output_directory> <processes>', fetch_instances = False)

    processes = int(args[3])
    config_path = pathlib.Path(args[1])
    instance_path = config_path.parents[0] / config['instances']
    output_directory = pathlib.Path(args[2])

    instance_list['partition'] = np.floor(np.arange(len(instance_list.index))*processes/len(instance_list.index)).astype(int)

    config_list = []
    for i, part in instance_list.groupby('partition'):
        instance_path = output_directory / (str(i)+'.dat')
        
        instance.write_instance_list(instance_path, part)
        parted_config = config
        parted_config['instances'] = str(instance_path.resolve())
        config_list.append(output_directory/(str(i)+'.json'))
        with config_list[-1].open('w') as f:
            json.dump(parted_config, f)

    with (output_directory/'config_list').open('w') as f:
        f.write('\n'.join([str(conf.resolve()) for conf in config_list]))
if __name__ == "__main__":
    partition()
