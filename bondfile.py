import pandas as pd
import numpy as np

# attempts to parse all files in dir as a bondfile and outputs bondfiles compatible with propane in outdir
# this renames spin indices such that there are no non-connected indices, removes duplicate couplers, and puts spin count and normalization at the top

def read_bondfile(filename):
    bonds = pd.read_csv(filename, delim_whitespace=True, header=None, names=['i','j','J_ij'], comment='#')

    indices = np.union1d(np.unique(bonds['i']), np.unique(bonds['j']))

    new_indices = np.vectorize(lambda i: np.where(indices==i)[0][0])

    bonds['i'] = list(new_indices(bonds['i']))
    bonds['j'] = list(new_indices(bonds['j']))

    bonds.loc[bonds['i'] > bonds['j'], ['j', 'i']] = bonds.loc[bonds['i'] > bonds['j'], ['i', 'j']].values

    bonds['field'] = bonds['i'] == bonds['j']
    bonds.sort_values(['field', 'i', 'j'], inplace=True)

    bonds.drop_duplicates(inplace=True)
    del bonds['field']
    return bonds

def write_bondfile(bonds, buffer):
    buffer.write(str(np.max([np.max(bonds['i']), np.max(bonds['j'])]) + 1) +' 1\n')
    bonds.to_csv(buffer, sep=' ', index=False, header=False)