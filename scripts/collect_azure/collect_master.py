import os

from tqdm import tqdm


def run_cmd(cmd):
    print(cmd)
    ret = os.system(cmd)
    if ret != 0:
        exit(ret)

def ssh_run(host, cmd):
    cmd = cmd.replace("\"", "\\\"")
    run_cmd(f"ssh -o StrictHostKeyChecking=no {host} \"{cmd}\"")


def ssh_tmux_run(host, cmd):
    cmd = f"tmux new-session -d \"{cmd}\""
    ssh_run(host, cmd)


n_tasks = 1577
n_machines = 20
tasks_per_machine = (n_tasks + n_machines - 1) // n_machines

if __name__ == "__main__":
    target = "llvm -mcpu=skylake-avx512 -model=platinum-8272"

    print(f"Tasks_per_machine: {tasks_per_machine}")

    for i in tqdm(range(0, n_machines)):
        host_name = f"azure-intel-avx512-{i:02d}"

        start_idx = i * tasks_per_machine
        end_idx = (i + 1) * tasks_per_machine

        ssh_run(host_name, "hostname")

        # fetch code
        ssh_run(host_name, "cd tvm-cost-model; git reset --hard 68a29876fe; git pull;")

        ## run collection
        worker_commond = "source ~/.bashrc; cd tvm-cost-model/scripts; "\
                         "PYTHONPATH=~/tvm-cost-model/python python3 collect_azure/collect_worker.py "\
                        f"--start-idx {i} --end-idx {n_tasks} --step-idx {n_machines} "\
                        f"--target '{target}'"
        ssh_tmux_run(host_name, worker_commond)

