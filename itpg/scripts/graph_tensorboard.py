from tensorboard.backend.event_processing import event_accumulator
import seaborn as sns
import re
from scipy.interpolate import make_interp_spline, BSpline
import numpy as np
import matplotlib.pyplot as plt

# Gear Images
event_files = [
    "/home/choudhue/PolicyGuide/results/policyguide/events.out.tfevents.1745677937.aisgpu4"
]



labels = ["rgb"]
for i, file in enumerate(event_files):
    event_acc = event_accumulator.EventAccumulator(file)
    event_acc.Reload()

    # Extract histlen from the file string
    # if re.search(r'hist(\d+)', file) == None:
    #     histlen = '0'
    # else:
    #     histlen = re.search(r'hist(\d+)', file).group(1)
    # print(histlen)

    # Get the scalar data from the event file
    tags = event_acc.Tags()['scalars']
    print(tags)
    data = {}
    for tag in tags:
        data[tag] = event_acc.Scalars(tag)

    # Plot the data using seaborn or matplotlib
    # sns.set_theme(style="darkgrid")

    tag = 'hp_metric'
    label = labels[i]

    scalar = data[tag]
    x = [s.step for s in scalar]
    y = [s.value for s in scalar]
    x = x[30:]
    y = y[30:]
    plt.plot(x, y, label=label)

y_min, y_max = min(y), 1
num_ticks = 10  # Increase this number for more granular ticks
yticks = np.linspace(y_min, y_max, num_ticks)
plt.yticks(yticks, [f"{tick:.2f}" for tick in yticks])
 
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.savefig('training_loss.png')