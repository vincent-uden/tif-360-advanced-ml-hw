import h5py
import numpy as np

if __name__ == "__main__":
    f = h5py.File("./cache/q_table/2023-03-29T10:33:07.485639.hdf5", "r")

    print(list(f.keys()))
    print(np.nonzero(f["q_table"][:,:] != 1000.0))
