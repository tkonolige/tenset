import tvm
from tvm.auto_scheduler.measure_record import RecordReader
import pickle
import h5py
import numpy as np
import argparse

from tokenize_tir import TIRTokenizer

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
    dset_programs = f.create_dataset("tir_programs", (10**4,), dtype=h5py.string_dtype(), chunks=(10**4,), maxshape=(None,), compression="gzip")
    vlen = h5py.vlen_dtype(np.dtype('float32'))
    dset_times = f.create_dataset("times", (10**4,), dtype=vlen, chunks=(10**4,), maxshape=(None,), compression="gzip")
    dset_mean_times = f.create_dataset("mean_times", (10**4,), dtype="float32", chunks=(10**4,), maxshape=(None,), compression="gzip")
    dset_targets = f.create_dataset("target", (10**4,), dtype=h5py.string_dtype(), chunks=(10**4,), maxshape=(None,), compression="gzip")

    offset = 0

    for filename in args.logs:

        mean_times = []
        times = []
        programs = []
        tokens = []
        targets = []

        rr = RecordReader(filename)
        for i, (inp, result) in enumerate(rr):
            task = all_tasks_mapping[inp.task.workload_key]
            sch, args = task.compute_dag.apply_steps_from_state(inp.state)
            mod = tvm.lower(sch, args)
            tir = mod["main"]
            programs.append(tir.astext())
            tks = TIRTokenizer().visit(tir)
            tks_ary = np.stack([np.array([x[0] for x in tks], dtype="int32"), np.array([x[1].value for x in tks], dtype="int32")])
            tokens.append(tks_ary)
            costs = np.array([x.value for x in list(result.costs)])
            times.append(costs)
            mean_times.append(np.mean(costs))
            targets.append(str(inp.task.target))

        n = len(programs)
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
            dset_tokens[offset+i,:] = x
        offset += n
