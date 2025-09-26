import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
        last_val = success_rates[-1] if success_rates else 0
        success_rates.extend([last_val] * (5 - len(success_rates)))

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

def plot_structured_bar_graph(data_path, n_samples=100):
    """
    Loads data and plots a structured bar graph with grouped and single columns.
    """
    with open(data_path, 'r') as f:
        data = json.load(f)['results']

    # Define the order and structure
    plot_order = [
        "Random",
        "ITPS",
        "RS",
        # "CALVIN Benchmark (Static+Wrist RGB)",
        "Language Benchmark",
        # "RobotUniView (SOTA)"
    ]
    
    # Colors for grouped bars and single bars
    group_colors = {'Point': '#1f77b4', 'Path': '#ff7f0e', 'Trajectory': '#2ca02c'}
    single_color = '#fdcfa0'

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Store data for plotting
    plot_data = {key: [] for key in plot_order}
    for item in data:
        gt = item['Guide Type']
        if gt in plot_data:
            plot_data[gt].append(item)

       # X-axis positions
    x_positions = np.arange(len(plot_order))
    bar_width = 0.2  # Width for each bar in a group
    group_width = bar_width * 3

    # --- Plotting ---
     # Random (single bar)
    random_data = plot_data['Random'][0]
    error = estimate_ci_from_success_rates(random_data['chained_sr'], n_samples)
    height = random_data['Avg Len']
    ax.bar(x_positions[0], height, yerr=error, color=single_color, width=group_width, capsize=5, label='_nolegend_')
    ax.text(x_positions[0], height + error, f'{height:.2f}', ha='center', va='bottom', fontsize=10)

    # ITPS (grouped bar)
    itps_modes = ['Point', 'Path', 'Trajectory']
    itps_data = sorted(plot_data['ITPS'], key=lambda x: itps_modes.index(x['Aff. Mode']))
    for i, mode in enumerate(itps_modes):
        item = itps_data[i]
        error = estimate_ci_from_success_rates(item['chained_sr'], n_samples)
        pos = x_positions[1] - group_width / 2 + bar_width / 2 + i * bar_width
        height = item['Avg Len']
        ax.bar(pos, height, yerr=error, color=group_colors[mode], width=bar_width, capsize=5, label=mode if x_positions[1] == 1 and i == 0 else '_nolegend_')
        ax.text(pos, height + error, f'{height:.2f}', ha='center', va='bottom', fontsize=10)

    # RS (grouped bar)
    rs_modes = ['Point', 'Path', 'Trajectory']
    rs_data = sorted(plot_data['RS'], key=lambda x: rs_modes.index(x['Aff. Mode']))
    for i, mode in enumerate(rs_modes):
        item = rs_data[i]
        error = estimate_ci_from_success_rates(item['chained_sr'], n_samples)
        pos = x_positions[2] - group_width / 2 + bar_width / 2 + i * bar_width
        height = item['Avg Len']
        ax.bar(pos, height, yerr=error, color=group_colors[mode], width=bar_width, capsize=5, label='_nolegend_')
        ax.text(pos, height + error, f'{height:.2f}', ha='center', va='bottom', fontsize=10)

    # Single bars for the rest
    for i in range(3, len(plot_order)):
        key = plot_order[i]
        item_data = plot_data[key][0]
        # if i == 3:  # CALVIN Benchmark
        #     error = estimate_ci_from_success_rates(item_data['chained_sr'], 1000)
        # else:
        error = estimate_ci_from_success_rates(item_data['chained_sr'], n_samples)
        height = item_data['Avg Len']
        ax.bar(x_positions[i], height, yerr=error, color=single_color, width=group_width, capsize=5, label='_nolegend_')
        ax.text(x_positions[i], height + error, f'{height:.2f}', ha='center', va='bottom', fontsize=10)

    # --- Formatting ---
    ax.set_ylabel('Average Sequence Length', fontsize=14)
    ax.set_xlabel('Method', fontsize=14)

    ax.set_title('Average Sequence Length for Methods using the Play Dataset', fontsize=16)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([label.replace(" (Static+Wrist RGB)", "") for label in plot_order], fontsize=12) #, rotation=20, ha="right"
    
    # Create custom legend
    legend_patches = [mpatches.Patch(color=color, label=mode) for mode, color in group_colors.items()]
    legend_patches.append(mpatches.Patch(color=single_color, label='No Affordance'))
    ax.legend(handles=legend_patches)
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    # plt.show()
    # Save the plot
    fig.savefig('FullT_graph_nobench.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'FullT_graph_nobench.png'.")


if __name__ == '__main__':
    json_file_path = '/home/choudhue/PolicyGuide/evaluations/fullT/total_results.json'
    plot_structured_bar_graph(json_file_path)