import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
import pandas as pd

def confidence_interval(p, n):
    return 1.96 * np.sqrt(p * (1 - p) / n)

# Path to the root directory
root_dir = "evaluations/fullT"

# Storage
seq_lens = []
run_labels = []
task_success = defaultdict(list)

# Loop through each run
for run_dir in os.listdir(root_dir):
    if "benchmark" in run_dir.lower():
        continue
    run_path = os.path.join(root_dir, run_dir)
    if not os.path.isdir(run_path):
        continue

    run_seq_lens = []

    for file in os.listdir(run_path):
        if not file.endswith(".json"):
            continue
        with open(os.path.join(run_path, file), "r") as f:
            data = json.load(f)
            if "100" not in data:
                continue
            d100 = data["100"]
            run_seq_lens.append(d100["avg_seq_len"])

            for task, res in d100["task_info"].items():
                success_rate = res["success"] / res["total"] if res["total"] > 0 else 0
                task_success[task].append((run_dir, success_rate))

    if run_seq_lens:
        seq_lens.append(run_seq_lens)
        run_labels.append(run_dir)

# Flatten data into a long-form DataFrame
task_records = []
task_means = {}

for task, entries in task_success.items():
    for run, rate in entries:
        task_records.append({"task": task, "run": run, "success_rate": rate})
    # Mean across all runs (for sorting)
    task_means[task] = np.mean([r[1] for r in entries if r[1] is not None])

# Convert to DataFrame
df = pd.DataFrame(task_records)
sorted_tasks = sorted(task_means.items(), key=lambda x: x[1], reverse=True)
task_order = [task for task, _ in sorted_tasks]

# Create the plot
plt.figure(figsize=(10, 12))
sns.set_style("whitegrid")

# Create a horizontal bar plot of the average success rate.
# Seaborn's barplot will automatically calculate the mean for each task.
# The error bars will show the standard deviation across runs.
ax = sns.barplot(x="success_rate", y="task", data=df, order=task_order, palette="viridis", errorbar="sd")

# Set plot titles and labels
ax.set_title("Average Task Success Rate Across All Runs", fontsize=16)
ax.set_xlabel("Average Success Rate", fontsize=12)
ax.set_ylabel("Task", fontsize=12)
ax.set_xlim(0, 1.0)

plt.tight_layout()
plt.show()