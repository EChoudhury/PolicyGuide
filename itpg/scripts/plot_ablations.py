import json
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
        "ITPS": ['#a6cee3', '#1f78b4', "#003e70"],  # Shades of blue
        "RS": ['#fdd0a2', '#fc8d3c', '#e5550d'],    # Shades of green
        "BENCH": ['#33a02c', '#b2df8a', '#fb9a99']  # Shades of red
    }

def confidence_interval(p, n):
    # Ensure p is within the valid range [0, 1] for this formula
    p = np.clip(p, 0, 1)
    return 1.96 * np.sqrt(p * (1 - p) / n)

# def confidence_interval(std_dev, n, confidence_level=0.95):
#     """
#     Calculates the confidence interval for a given standard deviation and sample size.
#     """
#     z_score = scipy.stats.norm.ppf((1 + confidence_level) / 2)
#     return z_score * (std_dev / np.sqrt(n))

def estimate_ci_from_success_rates(success_rates, n_samples):
        """
        Estimates the confidence interval for the mean sequence length.

        Args:
            success_rates (list): A list of 5 success probabilities [p1, p2, p3, p4, p5].
            n_samples (int): The number of sequences tested (e.g., 100).

        Returns:
            float: The margin of error for a 95% confidence interval.
        """
        if len(success_rates) != 5:
            raise ValueError("success_rates must be a list of 5 probabilities.")

        probs = []
        # P(L=1) = 1 - p1
        probs.append(1 - success_rates[0])
        # P(L=k) = p1*...*p(k-1)*(1-pk) for k=2,3,4
        p_cumulative = success_rates[0]
        for i in range(1, 4):
            probs.append(p_cumulative * (1 - success_rates[i]))
            p_cumulative *= success_rates[i]
        # P(L=5) = p1*p2*p3*p4
        probs.append(p_cumulative)

        lengths = np.arange(1, 6)
        
        # Calculate theoretical mean and variance from the distribution
        mean_len_sq = np.sum([(l**2) * p for l, p in zip(lengths, probs)])
        mean_len = np.sum([l * p for l, p in zip(lengths, probs)])
        variance = mean_len_sq - (mean_len**2)
        
        # Estimate standard deviation
        std_dev = np.sqrt(variance)
        
        # Calculate 95% confidence interval margin of error
        z_score = 1.96
        margin_of_error = z_score * (std_dev / np.sqrt(n_samples))
        
        return margin_of_error


def plot_affordance_mode_data(json_file_path):
    """
    Loads data from a JSON file and plots the average length for each method 
    across different affordance modes.
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    methods = ["ITPS", "RS"]
    sub_methods = ["Point", "Path", "Trajectory"]
    
    num_method_groups = len(methods) * len(sub_methods)
    width = 0.14

    # --- Plot for Distance ---
    dist_modes = ["Dist 0.5", "Dist 0.25", "Dist 0.125"]
    plot_data_dist = {f"{m}-{sm}": [] for m in methods for sm in sub_methods}

    for dist_val in sorted(data["Distance"].keys(), reverse=True):
        for method in methods:
            for sub_method in sub_methods:
                avg_len = data["Distance"][dist_val][method][sub_method]["avg_len"]
                plot_data_dist[f"{method}-{sub_method}"].append(avg_len)

    x_dist = np.arange(len(dist_modes))
    fig_dist, ax_dist = plt.subplots(figsize=(14, 8))
    
    i = 0
    for method in methods:
        for sub_method_idx, sub_method in enumerate(sub_methods):
            method_name = f"{method}-{sub_method}"
            values = np.array(plot_data_dist[method_name])
            
            offset = width * (i - (num_method_groups - 1) / 2)
            bar_positions = x_dist + offset
            color = COLORS[method][sub_method_idx]

            # Calculate confidence intervals
            n = 100 # As specified
            y_err = confidence_interval(values, n)

            rects = ax_dist.bar(bar_positions, values, width, yerr=y_err, capsize=4, label=method_name, color=color)
            ax_dist.bar_label(rects, padding=3, fmt='%.2f', fontsize=8)
            ax_dist.plot(bar_positions, values, marker='o', linestyle='--', color=color, alpha=0.7)
            i += 1

    ax_dist.set_ylabel('Average Length')
    ax_dist.set_title('Method Performance Across Distance Modes')
    ax_dist.set_xticks(x_dist)
    ax_dist.set_xticklabels(dist_modes)
    ax_dist.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax_dist.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
    fig_dist.tight_layout()
    plt.savefig("ablations_plot_distance.png")

    # --- Plot for Time ---
    time_modes = ["20 Steps", "30 Steps"]
    plot_data_time = {f"{m}-{sm}": [] for m in methods for sm in sub_methods}

    for time_val in sorted(data["Time"].keys()):
        for method in methods:
            for sub_method in sub_methods:
                avg_len = data["Time"][time_val][method][sub_method]["avg_len"]
                plot_data_time[f"{method}-{sub_method}"].append(avg_len)

    x_time = np.arange(len(time_modes))
    fig_time, ax_time = plt.subplots(figsize=(14, 8))

    i = 0
    for method in methods:
        for sub_method_idx, sub_method in enumerate(sub_methods):
            method_name = f"{method}-{sub_method}"
            values = np.array(plot_data_time[method_name])
            
            offset = width * (i - (num_method_groups - 1) / 2)
            bar_positions = x_time + offset
            color = COLORS[method][sub_method_idx]

            # Calculate confidence intervals
            n = 100 # As specified
            y_err = confidence_interval(values, n)

            rects = ax_time.bar(bar_positions, values, width, yerr=y_err, capsize=4, label=method_name, color=color)
            ax_time.bar_label(rects, padding=3, fmt='%.2f', fontsize=12)
            # The line plot doesn't typically show error bars, so it remains unchanged.
            ax_time.plot(bar_positions, values, marker='o', linestyle='--', color=color, alpha=0.7)
            i += 1

    ax_time.set_ylabel('Average Length')
    ax_time.set_title('Method Performance Across Time Modes')
    ax_time.set_xticks(x_time)
    ax_time.set_xticklabels(time_modes)
    ax_time.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax_time.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
    fig_time.tight_layout()
    plt.savefig("ablations_plot_time.png")
    
    plt.show()


def plot_affordance_mode_data_new_ci(json_file_path):
    """
    Loads data from a JSON file and plots the average length for each method 
    across different affordance modes.
    """
    plt.rcParams.update({'font.size': 16})
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    methods = ["ITPS", "RS"]
    sub_methods = ["Point", "Path", "Trajectory"]
    
    num_method_groups = len(methods) * len(sub_methods)
    width = 0.14

    # --- Plot for Distance ---
    dist_modes = ["0.5", "0.25", "0.125"]
    plot_data_dist = {f"{m}-{sm}": [] for m in methods for sm in sub_methods}
    plot_data_dist_sr = {f"{m}-{sm}": [] for m in methods for sm in sub_methods}

    for dist_val in sorted(data["Distance"].keys(), reverse=True):
        for method in methods:
            for sub_method in sub_methods:
                avg_len = data["Distance"][dist_val][method][sub_method]["avg_len"]
                plot_data_dist[f"{method}-{sub_method}"].append(avg_len)
                success_rates = data["Distance"][dist_val][method][sub_method]["chain_sr"]
                plot_data_dist_sr[f"{method}-{sub_method}"].append(success_rates)

    x_dist = np.arange(len(dist_modes))
    fig_dist, ax_dist = plt.subplots(figsize=(14, 8))
    
    i = 0
    for method in methods:
        for sub_method_idx, sub_method in enumerate(sub_methods):
            method_name = f"{method}-{sub_method}"
            values = np.array(plot_data_dist[method_name])
            
            offset = width * (i - (num_method_groups - 1) / 2)
            bar_positions = x_dist + offset
            color = COLORS[method][sub_method_idx]

            # Calculate confidence intervals
            n = 100 # As specified
            success_rates_list = plot_data_dist_sr[method_name]
            y_err = [estimate_ci_from_success_rates(sr, n) for sr in success_rates_list]

            rects = ax_dist.bar(bar_positions, values, width, yerr=y_err, capsize=4, label=method_name, color=color)
            ax_dist.bar_label(rects, padding=3, fmt='%.2f', fontsize=10)
            # ax_dist.plot(bar_positions, values, marker='o', linestyle='--', color=color, alpha=0.7)
            i += 1

    ax_dist.set_ylabel('Average Length')
    ax_dist.set_xlabel('Distance (m)')
    ax_dist.set_title('Method Performance Across Distance Modes')
    ax_dist.set_xticks(x_dist)
    ax_dist.set_xticklabels(dist_modes)
    ax_dist.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax_dist.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
    fig_dist.tight_layout()
    plt.savefig("ablations_plot_distance_ci_nolines_colorchange.png")

    # --- Plot for Time ---
    time_modes = ["20 Steps", "30 Steps"]
    plot_data_time = {f"{m}-{sm}": [] for m in methods for sm in sub_methods}
    plot_data_time_sr = {f"{m}-{sm}": [] for m in methods for sm in sub_methods}

    for time_val in sorted(data["Time"].keys()):
        for method in methods:
            for sub_method in sub_methods:
                avg_len = data["Time"][time_val][method][sub_method]["avg_len"]
                plot_data_time[f"{method}-{sub_method}"].append(avg_len)
                success_rates = data["Time"][time_val][method][sub_method]["chain_sr"]
                plot_data_time_sr[f"{method}-{sub_method}"].append(success_rates)

    x_time = np.arange(len(time_modes))
    fig_time, ax_time = plt.subplots(figsize=(14, 8))

    i = 0
    for method in methods:
        for sub_method_idx, sub_method in enumerate(sub_methods):
            method_name = f"{method}-{sub_method}"
            values = np.array(plot_data_time[method_name])
            
            offset = width * (i - (num_method_groups - 1) / 2)
            bar_positions = x_time + offset
            color = COLORS[method][sub_method_idx]

            # Calculate confidence intervals
            n = 100 # As specified
            success_rates_list = plot_data_time_sr[method_name]
            y_err = [estimate_ci_from_success_rates(sr, n) for sr in success_rates_list]

            rects = ax_time.bar(bar_positions, values, width, yerr=y_err, capsize=4, label=method_name, color=color)
            ax_time.bar_label(rects, padding=3, fmt='%.2f', fontsize=10)
            # The line plot doesn't typically show error bars, so it remains unchanged.
            # ax_time.plot(bar_positions, values, marker='o', linestyle='--', color=color, alpha=0.7)
            i += 1

    ax_time.set_ylabel('Average Length')
    ax_time.set_xlabel('Duration (Steps)')
    ax_time.set_title('Method Performance Across Time Modes')
    ax_time.set_xticks(x_time)
    ax_time.set_xticklabels(time_modes)
    ax_time.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax_time.yaxis.grid(True, linestyle='--', which='both', color='grey', alpha=.25)
    fig_time.tight_layout()
    plt.savefig("ablations_plot_time_ci_nolines_colorchange.png")
    
    # plt.show()


def plot_dataset_size_data(data_path, save_path=None):
    """
    Plots the dataset size ablation results from a JSON file.
    """
    plt.rcParams.update({'font.size': 12})
    with open(data_path, "r") as f:
        data = json.load(f)

    labels = list(data.keys())
    itps_avg_len = [data[label]["itps_traj"]["avg_len"] for label in labels]
    rs_avg_len = [data[label]["rs_traj"]["avg_len"] for label in labels]
    bench_avg_len = [data[label]["Bench"]["avg_len"] for label in labels]

    itps_success_rates = [data[label]["itps_traj"]["success_rates"] for label in labels] # Placeholder
    rs_success_rates = [data[label]["rs_traj"]["success_rates"] for label in labels]   # Placeholder
    bench_success_rates = [data[label]["Bench"]["success_rates"] for label in labels]# Placeholder

    n = 100 # Your sample size
    itps_ci = [estimate_ci_from_success_rates(rates, n) for rates in itps_success_rates]
    rs_ci = [estimate_ci_from_success_rates(rates, n) for rates in rs_success_rates]
    bench_ci = [estimate_ci_from_success_rates(rates, n) for rates in bench_success_rates]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(
        x - width,
        itps_avg_len,
        width,
        yerr=itps_ci,
        label="ITPS Trajectory",
        color=COLORS["ITPS"][1],
        capsize=5,
    )
    rects2 = ax.bar(
        x,
        rs_avg_len,
        width,
        yerr=rs_ci,
        label="RS Trajectory",
        color=COLORS["RS"][0],
        capsize=5,
    )
    rects3 = ax.bar(
        x + width,
        bench_avg_len,
        width,
        yerr=bench_ci,
        label="Benchmark",
        color=COLORS["BENCH"][1],
        capsize=5,
    )

    # Add lines to connect the bars
    # ax.plot(x - width, itps_avg_len, color=COLORS["ITPS"][1], marker="o", linestyle='--')
    # ax.plot(x, rs_avg_len, color=COLORS["RS"][1], marker="o", linestyle='--')
    # ax.plot(x + width, bench_avg_len, color=COLORS["BENCH"][1], marker="o", linestyle='--')
    
    # ax.xaxis.grid(True, linestyle='--', which='both', color='grey', alpha=.25)

    ax.set_ylabel("Average Sequence Length")
    ax.set_xlabel("Demonstrations per Task")
    ax.set_title("Performance by Dataset Size")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()


    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.25, zorder=0)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_rollout_ablation_shaded(json_path):
    """
    Plots the rollout ablation results from a JSON file.
    """
    plt.rcParams.update({'font.size': 12})
    with open(json_path, "r") as f:
        data = json.load(f)

    ablation_data = data["FullT"]
    itps_data = ablation_data["itps_traj"]
    rs_data = ablation_data["rs_traj"]

    labels = sorted(itps_data.keys(), key=lambda x: int(x.replace("k", "")))
    itps_avg_len = np.array([itps_data[k]["Avg Len"] for k in labels])
    rs_avg_len = np.array([rs_data[k]["Avg Len"] for k in labels])

    itps_chain_sr = [itps_data[k]["chain_sr"] for k in labels]
    rs_chain_sr = [rs_data[k]["chain_sr"] for k in labels]

    n = 100  # Sample size
    itps_ci = np.array([estimate_ci_from_success_rates(rates, n) for rates in itps_chain_sr])
    rs_ci = np.array([estimate_ci_from_success_rates(rates, n) for rates in rs_chain_sr])

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot ITPS data
    ax.plot(
        x,
        itps_avg_len,
        marker="o",
        linestyle="-",
        label="ITPS Trajectory",
        color=COLORS["ITPS"][2],
    )
    ax.fill_between(
        x, itps_avg_len - itps_ci, itps_avg_len + itps_ci, color=COLORS["ITPS"][2], alpha=0.2
    )

    # Plot RS data
    ax.plot(
        x,
        rs_avg_len,
        marker="o",
        linestyle="-",
        label="RS Trajectory",
        color=COLORS["RS"][0],
    )
    ax.fill_between(
        x, rs_avg_len - rs_ci, rs_avg_len + rs_ci, color=COLORS["RS"][0], alpha=0.2
    )

    ax.set_ylabel("Average Length")
    ax.set_xlabel("Number of Training Steps")
    ax.set_title("Average Sequence Length Across Training Duration")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, which="major", linestyle="--", linewidth=0.5)

    fig.tight_layout()
    plt.savefig("rollout_ablation_shaded.png")
    # plt.show()


def plot_rollout_ablation(json_path):
    """
    Plots the rollout ablation results from a JSON file.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    ablation_data = data["FullT"]
    itps_data = ablation_data["itps_traj"]
    rs_data = ablation_data["rs_traj"]

    labels = sorted(itps_data.keys(), key=lambda x: int(x.replace("k", "")))
    itps_avg_len = [itps_data[k]["Avg Len"] for k in labels]
    rs_avg_len = [rs_data[k]["Avg Len"] for k in labels]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        x,
        itps_avg_len,
        marker="o",
        linestyle="-",
        label="ITPS Trajectory",
        color=COLORS["ITPS"][1],
    )
    ax.plot(
        x,
        rs_avg_len,
        marker="o",
        linestyle="-",
        label="RS Trajectory",
        color=COLORS["RS"][0],
    )

    ax.set_ylabel("Average Length")
    ax.set_xlabel("Number of Training Steps")
    ax.set_title("Average Sequence Length Across Training Duration")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    fig.tight_layout()
    plt.savefig("rollout_ablation.png")
    plt.show()


def plot_rollout_ablation_ci(json_path, save_path=None):
    """
    Plots the rollout ablation results from a JSON file.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    ablation_data = data["FullT"]
    itps_data = ablation_data["itps_traj"]
    rs_data = ablation_data["rs_traj"]

    labels = sorted(itps_data.keys(), key=lambda x: int(x.replace("k", "")))
    itps_avg_len = [itps_data[k]["Avg Len"] for k in labels]
    rs_avg_len = [rs_data[k]["Avg Len"] for k in labels]

    itps_chain_sr = [itps_data[k]["chain_sr"] for k in labels]
    rs_chain_sr = [rs_data[k]["chain_sr"] for k in labels]

    n = 100  # Sample size
    itps_ci = [estimate_ci_from_success_rates(rates, n) for rates in itps_chain_sr]
    rs_ci = [estimate_ci_from_success_rates(rates, n) for rates in rs_chain_sr]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(
        x,
        itps_avg_len,
        yerr=itps_ci,
        marker="o",
        linestyle="-",
        label="ITPS Trajectory",
        color=COLORS["ITPS"][2],
        capsize=5,
    )
    ax.errorbar(
        x,
        rs_avg_len,
        yerr=rs_ci,
        marker="o",
        linestyle="-",
        label="RS Trajectory",
        color=COLORS["RS"][0],
        capsize=5,
    )

    ax.set_ylabel("Average Length")
    ax.set_xlabel("Number of Training Steps")
    ax.set_title("Average Sequence Length Across Training Duration")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    fig.tight_layout()

    plt.savefig("rollout_ablations.png")

    plt.show()


if __name__ == '__main__':
    # The user mentioned the file is at this path.
    # Make sure this path is correct or adjust as needed.
    json_path = '/home/choudhue/PolicyGuide/evaluations/ablations/affordance_ablation_results.json'
    plot_affordance_mode_data_new_ci(json_path)

    # plot_dataset_size_data(
    #     "/home/choudhue/PolicyGuide/evaluations/ablations/dataset_size_ablation_results.json",
    #     save_path="dataset_size_ablation_bargraph.png",
    # )

    # plot_rollout_ablation_shaded("/home/choudhue/PolicyGuide/evaluations/ablations/rollout_ablation_results.json")


    