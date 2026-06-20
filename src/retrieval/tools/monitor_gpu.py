import argparse
import csv
import datetime as dt
import subprocess
import time
from pathlib import Path


def run_query(args_list):
    result = subprocess.run(
        args_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def query_gpu_rows():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    rows = []
    for line in run_query(cmd):
        parts = [x.strip() for x in line.split(",")]
        rows.append(
            {
                "gpu_index": parts[0],
                "gpu_name": parts[1],
                "gpu_util": parts[2],
                "mem_util": parts[3],
                "mem_used_mb": parts[4],
                "mem_total_mb": parts[5],
                "power_w": parts[6],
                "temp_c": parts[7],
            }
        )
    return rows


def query_compute_rows():
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,gpu_uuid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        lines = run_query(cmd)
    except subprocess.CalledProcessError:
        return []
    rows = []
    for line in lines:
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 3:
            rows.append(
                {
                    "pid": parts[0],
                    "gpu_uuid": parts[1],
                    "proc_mem_mb": parts[2],
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=0.0, help="0 means run until interrupted")
    parser.add_argument("--pid", type=int, default=None, help="optional process id to highlight")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp",
        "gpu_index",
        "gpu_name",
        "gpu_util",
        "mem_util",
        "mem_used_mb",
        "mem_total_mb",
        "power_w",
        "temp_c",
        "pid",
        "proc_mem_mb",
        "target_pid_match",
    ]

    start = time.time()
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()
        while True:
            now = dt.datetime.now().isoformat(timespec="seconds")
            gpu_rows = query_gpu_rows()
            proc_rows = query_compute_rows()

            if not proc_rows:
                for gpu_row in gpu_rows:
                    writer.writerow(
                        {
                            "timestamp": now,
                            **gpu_row,
                            "pid": "",
                            "proc_mem_mb": "",
                            "target_pid_match": "",
                        }
                    )
            else:
                for gpu_row in gpu_rows:
                    wrote = False
                    for proc_row in proc_rows:
                        writer.writerow(
                            {
                                "timestamp": now,
                                **gpu_row,
                                "pid": proc_row["pid"],
                                "proc_mem_mb": proc_row["proc_mem_mb"],
                                "target_pid_match": int(args.pid is not None and int(proc_row["pid"]) == args.pid),
                            }
                        )
                        wrote = True
                    if not wrote:
                        writer.writerow(
                            {
                                "timestamp": now,
                                **gpu_row,
                                "pid": "",
                                "proc_mem_mb": "",
                                "target_pid_match": "",
                            }
                        )

            f.flush()

            if args.duration > 0 and (time.time() - start) >= args.duration:
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
