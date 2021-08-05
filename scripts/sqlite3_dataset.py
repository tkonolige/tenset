import sqlite3
import tvm
from tvm.auto_scheduler.measure_record import RecordReader
import pickle
import os
import glob
import pathlib

import torch
from torch.utils.data import Dataset
import numpy as np


class TIRPrograms(Dataset):
    def __init__(self, db_file, all_tasks_file):
        self.all_tasks = pickle.load(open(all_tasks_file, "rb"))
        self.all_tasks_mapping = {x.workload_key: x for x in self.all_tasks}
        self.con = sqlite3.connect(db_file)
        self.cur = self.con.cursor()
        self.len = next(self.cur.execute("select count(*) from measure_records where train=TRUE"))[
            0
        ]
        self.read_record = tvm.get_global_func("auto_scheduler.ReadMeasureRecord")
        self.lower_for_feature_extraction = tvm.get_global_func("auto_scheduler.LowerForFeatureExtraction")

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        target, workload_key, json = next(
            self.cur.execute(
                "select target, workload_key, json from measure_records where train=TRUE limit 1 offset ?",
                (idx,),
            )
        )
        inp, result = self.read_record(json)
        task = self.all_tasks_mapping[workload_key]
        tir = self.lower_for_feature_extraction(task, inp.state)
        sch, args = task.compute_dag.apply_steps_from_state(inp.state)
        try:
            with tvm.target.Target(target):
                mod = tvm.driver.lower(sch, args)
                print(tir)
                print("-"*80)
                print(mod["main"])
                print("="*80)
        except:
            pass
        costs = np.array([x.value for x in list(result.costs)])
        return (target, tir), np.mean(costs)

    def iterate(self):
        for target, workload_key, json in self.cur.execute(
            "select target, workload_key, json from measure_records where train=TRUE order by sample_id",
        ):
            inp, result = self.read_record(json)
            costs = np.array([x.value for x in list(result.costs)])
            cost = np.mean(costs)
            if cost >= 1000000000.0:
                continue
            task = self.all_tasks_mapping[workload_key]
            tir = self.lower_for_feature_extraction(task, inp.state)
            sch, args = task.compute_dag.apply_steps_from_state(inp.state)
            try:
                with tvm.target.Target(target):
                    mod = tvm.driver.lower(sch, args)
                    print(tir)
                    print("-"*80)
                    print(mod["main"])
                    print("="*80)
            except:
                pass
            yield (target, tir), cost


if __name__ == "__main__":
    read_record = tvm.get_global_func("auto_scheduler.ReadMeasureRecord")

    all_tasks = pickle.load(open("dataset_cpu/network_info/all_tasks.pkl", "rb"))
    all_tasks_mapping = {x.workload_key: x for x in all_tasks}

    model_mapping = {}
    for f in glob.glob("dataset_cpu/network_info/*.task.pkl"):
        tasks, unknown = pickle.load(open(f, "rb"))
        for x in tasks:
            model_mapping[x.workload_key] = os.path.basename(f)[:-9]  # strip ".task.pkl"

    con = sqlite3.connect("dataset_cpu.db")
    cur = con.cursor()

    # Create table
    cur.execute(
        """CREATE TABLE IF NOT EXISTS measure_records
                   (machine text, target text, model text, workload_key text, json text, train bool)"""
    )

    for filename in glob.glob("dataset_cpu/measure_records/**/*.json", recursive=True):
        print(filename)
        machine = pathlib.Path(filename).parts[2]
        with open(filename) as f:
            for line in f:
                inp, results = read_record(line)
                target = inp.task.target
                workload_key = inp.task.workload_key
                model = model_mapping[workload_key]
                cur.execute(
                    "INSERT INTO measure_records VALUES (?, ?, ?, ?, ?, ?)",
                    (machine, str(target), model, workload_key, line, True),
                )
        # Save (commit) the changes
        con.commit()

    # We can also close the connection if we are done with it.
    # Just be sure any changes have been committed or they will be lost.
    con.close()
