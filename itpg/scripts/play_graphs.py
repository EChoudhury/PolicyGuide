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

task_groups = {
    "ST": [
        "open_drawer",
        "move_slider_left",
        "close_drawer",
        "turn_on_lightbulb",
        "turn_off_lightbulb",
        "move_slider_right",
        "turn_on_led",
        "turn_off_led",
        "rotate_red_block_right",
        "rotate_blue_block_right",
        "rotate_pink_block_right",
        "unstack_block"
    ],
    "3T": [
        "open_drawer",
        "turn_on_lightbulb",
        "turn_on_led",
    ]
}

# All tasks from the original grouping to define the scope of "FullT"
all_tasks = [
    "open_drawer", "close_drawer", "push_into_drawer", "lift_blue_block_drawer", "lift_red_block_drawer", "lift_pink_block_drawer", "place_in_drawer",
    "move_slider_left", "move_slider_right", "lift_blue_block_slider", "lift_red_block_slider", "lift_pink_block_slider", "place_in_slider",
    "turn_on_lightbulb", "turn_off_lightbulb",
    "turn_on_led", "turn_off_led",
    "lift_pink_block_table", "lift_blue_block_table", "lift_red_block_table",
    "push_pink_block_right", "push_blue_block_left", "push_red_block_left", "push_pink_block_left", "push_blue_block_right", "push_red_block_right",
    "rotate_red_block_right", "rotate_red_block_left", "rotate_blue_block_right", "rotate_blue_block_left", "rotate_pink_block_right", "rotate_pink_block_left",
    "stack_block", "unstack_block"
]
# The rest of the tasks are FullT
st_and_3t_tasks = set(task_groups["ST"] + task_groups["3T"])
full_t_tasks = [task for task in all_tasks if task not in st_and_3t_tasks]
task_groups["FullT"] = full_t_tasks

task_to_group = {}
# Assign groups, giving precedence to ST, then 3T, then FullT
for task in all_tasks:
    if task in task_groups["3T"]:
        task_to_group[task] = "3T"
    elif task in task_groups["ST"]:
        task_to_group[task] = "ST"
    else:
        task_to_group[task] = "FullT"

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
    group = task_to_group.get(task, "FullT") # Default to FullT if not found
    for run, rate in entries:
        task_records.append({"task": task, "run": run, "success_rate": rate, "group": group})
    # Mean across all runs (for sorting)
    task_means[task] = np.mean([r[1] for r in entries if r[1] is not None])

df = pd.DataFrame(task_records)

# Sort tasks by group and then by success rate
group_order = ["3T", "ST", "FullT"]
df['group'] = pd.Categorical(df['group'], categories=group_order, ordered=True)
df['mean_rate'] = df.groupby('task')['success_rate'].transform('mean')
df = df.sort_values(by=['group', 'mean_rate'], ascending=[True, False])

# Create a separate plot for each group
for group in group_order:
    group_df = df[df['group'] == group].copy()
    if group_df.empty:
        continue

    # Sort tasks within the group by success rate
    task_order = group_df.groupby('task')['mean_rate'].mean().sort_values(ascending=False).index

    # Determine figure size based on number of tasks
    num_tasks = len(task_order)
    fig_height = max(6, num_tasks * 0.5)
    plt.figure(figsize=(10, fig_height))

    ax = sns.barplot(x="success_rate", y="task", data=group_df, order=task_order, color="#1f77b4", ci=None)

    # Set plot titles and labels
    ax.set_title(f"Average Task Success Rate for {group} Tasks", fontsize=16)
    ax.set_xlabel("Average Success Rate", fontsize=12)
    ax.set_ylabel("Task", fontsize=12)
    ax.set_xlim(0, 1.0)

    plt.tight_layout()
    plt.savefig(f"task_distribution_{group}.png")
    plt.close() 