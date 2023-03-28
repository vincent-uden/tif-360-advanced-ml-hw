import h5py
import numpy as np

if __name__ == "__main__":
    f = h5py.File("test.hdf5", "a")
    try:
        dset = f.create_dataset("q_table", (100, 10), dtype="f")
    except ValueError:
        dset = f["q_table"]

    dset[:, :] = np.ones((100, 10))[:, :]

    print(list(f.keys()))
    print(f["q_table"][:, :])
