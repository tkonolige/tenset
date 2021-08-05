import tvm
from tvm.auto_scheduler.measure_record import RecordReader
import pickle
import h5py
import numpy as np
import argparse
import glob
import gc
from itertools import count, repeat
from more_itertools import consume
from multiprocessing import Pool

import torch
from torch.utils.data import Dataset


def process(x):
    tokenize = tvm.get_global_func("tir.analysis.tokenize")

    filename, i = x
    rr = iter(RecordReader(filename))
    consume(rr,i)
    inp, result = next(rr)

    target = inp.task.target
    task = all_tasks_mapping[inp.task.workload_key]
    sch, args = task.compute_dag.apply_steps_from_state(inp.state)
    try:
        with target:
            mod = tvm.driver.lower(sch, args)
    except:
        return -1, 0, 0, 0
    tir = mod["main"]
    tks = tokenize(tir).asnumpy()
    costs = np.array([x.value for x in list(result.costs)])
    return np.mean(costs), costs, str(target), tks

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


class TIRPrograms(Dataset):
    def __init__(self, all_tasks_file, directory):
        self.all_tasks = pickle.load(open(all_tasks_file, "rb"))
        self.all_tasks_mapping = {x.workload_key:x for x in self.all_tasks}
        self.files = glob.glob(f"{directory}/**/*.json", recursive=True)
        self.file_sizes = [file_len(file) for file in self.files]
        self.length = sum(self.file_sizes)
        self.read_record = tvm.get_global_func("auto_scheduler.ReadMeasureRecord")
        self.tokenize = tvm.get_global_func("tir.analysis.tokenize")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # look for which file the idx is in
        i = idx
        file_idx = 0
        while i >= 0:
            if i > self.file_sizes[i]:
                i = i - self.file_sizes[i]
                file_idx += 1
            else:
                break

        with open(self.files[file_idx]) as f:
            for j, l in enumerate(f):
                if j == i:
                    inp, result = self.read_record(l)
                    target = inp.task.target
                    task = self.all_tasks_mapping[inp.task.workload_key]
                    sch, args = task.compute_dag.apply_steps_from_state(inp.state)
                    try:
                        with target:
                            mod = tvm.driver.lower(sch, args)
                    except:
                        return 1e10, [], "invalid", -1
                    tir = mod["main"]
                    tks = self.tokenize(tir).asnumpy()
                    print(tir)
                    print(tks.shape[0])
                    costs = np.array([x.value for x in list(result.costs)])
                    return (target, tir), np.mean(costs)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", type=str)
    parser.add_argument("--out-file", type=str, default='tir_dataset.h5')
    parser.add_argument("--all-tasks", type=str)

    args = parser.parse_args()

    if args.all_tasks is None:
        raise RuntimeError("You must specify --all-tasks")

    all_tasks = pickle.load(open(args.all_tasks, "rb"))

    all_tasks_mapping = {x.workload_key:x for x in all_tasks}

    with h5py.File(open(args.out_file, "w+b"), "a") as f:
        vlen_int = h5py.vlen_dtype(np.dtype('int32'))
        dset_tokens = f.create_dataset("tokens", (10**4,2), dtype=vlen_int, chunks=(10**4,2), maxshape=(None,2), compression="gzip")
        vlen = h5py.vlen_dtype(np.dtype('float32'))
        dset_times = f.create_dataset("times", (10**4,), dtype=vlen, chunks=(10**4,), maxshape=(None,), compression="gzip")
        dset_mean_times = f.create_dataset("mean_times", (10**4,), dtype="float32", chunks=(10**4,), maxshape=(None,), compression="gzip")
        dset_targets = f.create_dataset("target", (10**4,), dtype=h5py.string_dtype(), chunks=(10**4,), maxshape=(None,), compression="gzip")

        offset = dset_tokens.shape[0]

        with Pool() as p:
            for filepath in args.logs:
                for filename in glob.glob(f"{filepath}/**/*.json", recursive=True):
                    print(filename)

                    r = p.map(process, zip(repeat(filename), range(file_len(filename))))

                    mean_times = [x[0] for x in r if x[0] > 0]
                    times = [x[1] for x in r if x[0] > 0]
                    targets = [x[2] for x in r if x[0] > 0]
                    tokens = [x[3] for x in r if x[0] > 0]

                    n = len(tokens)
                    new_size = (offset+n, )
                    # dset_programs.resize(new_size)
                    dset_times.resize(new_size)
                    dset_mean_times.resize(new_size)
                    dset_targets.resize(new_size)
                    dset_tokens.resize((new_size[0],2))

                    # dset_programs[offset:offset+n] = programs
                    dset_times[offset:offset+n] = times
                    dset_mean_times[offset:offset+n] = mean_times
                    dset_targets[offset:offset+n] = targets
                    for i, x in enumerate(tokens):
                        dset_tokens[offset+i,0] = x[:,0]
                        dset_tokens[offset+i,1] = x[:,1]
                    offset += n

                    del tokens
                    del targets
                    del mean_times
                    del times
                    gc.collect()
