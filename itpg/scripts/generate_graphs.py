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

# Storage for plotting
seq_lens = []          # average sequence lengths for "100"
run_labels = []        # subdirectory names (experiment labels)
task_success = defaultdict(list)  # maps task names to a list of success rates

# Loop through each run
for run_dir in os.listdir(root_dir):
    # if "benchmark" in run_dir.lower():
    #     continue  # Skip benchmark runs
    run_path = os.path.join(root_dir, run_dir)
    if not os.path.isdir(run_path):
        continue

    run_seq_lens = []

    for file in os.listdir(run_path):
        if not file.endswith(".json"):
            continue
        with open(os.path.join(run_path, file), "r") as f:
            data = json.load(f)
            # Only use the "100" entry
            if "200" not in data:
                continue
            d100 = data["200"]
            run_seq_lens.append(d100["avg_seq_len"])

            # Parse task success rates
            for task, res in d100["task_info"].items():
                success_rate = res["success"] / res["total"] if res["total"] > 0 else 0
                task_success[task].append(success_rate)

    if run_seq_lens:
        seq_lens.append(run_seq_lens)
        run_labels.append(run_dir)

# Calculate means and 95% confidence intervals
means = [np.mean(lengths) for lengths in seq_lens]
cis = [confidence_interval(lengths[0], 200) for lengths in seq_lens]
# cis = [1.96 * np.std(lengths) / np.sqrt(len(lengths)) for lengths in seq_lens]

# Define the desired order for the second part of the label
order_map = {"point": 0, "path": 1, "trajectory": 2}

# Custom sorting key
def custom_sort_key(item):
    label = item[0]
    parts = label.split('_', 1)
    method = parts[0]
    task = parts[1] if len(parts) > 1 else ""
    return (method, order_map.get(task, 99)) # Use a large number for any other tasks

# Sort by the custom key
sorted_data = sorted(zip(run_labels, means, cis), key=custom_sort_key)
sorted_labels, sorted_means, sorted_cis = zip(*sorted_data)

# Define colors for each method
color_map = {"benchmark": "salmon", "itps": "skyblue", "rs": "lightgreen"}
bar_colors = [color_map.get(label.split('_')[0], 'gray') for label in sorted_labels]

# Plot
plt.figure(figsize=(10, 6))
plt.bar(sorted_labels, sorted_means, yerr=sorted_cis, capsize=5, color=bar_colors, edgecolor='black')

# Create custom legend
import matplotlib.patches as mpatches
legend_patches = [mpatches.Patch(color=color, label=method) for method, color in color_map.items()]
plt.legend(handles=legend_patches, title="Method")

plt.ylabel("Average Sequence Length")
plt.xlabel("Run")
plt.title("Average Sequence Lengths Across Runs (100) — Grouped by Method")
plt.xticks(rotation=45, ha="right")
plt.grid(axis='y')
plt.tight_layout()
plt.show()