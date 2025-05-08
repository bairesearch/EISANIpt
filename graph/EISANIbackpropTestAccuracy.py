import matplotlib.pyplot as plt
import numpy as np

def subset(arr, subsetMask):
	arr = np.array(arr)[np.array(subsetMask)]
	return arr
	
graphedDatasets = [False, False, True, True, True, True, True, True, True, True, False, False, True] 	#tested datasets

y1_values = [0.0, 0.0, 62.86, 70.17, 90.15, 58.13, 91.41, 61.23, 85.54, 85.77, 92.12, 66.85, 95.45]
y2_values = [0.0, 0.0, 58.67, 69.97, 91.80, 51.87, 91.60, 55.91, 88.45, 81.68, 0.0, 0.0, 86.35]

group_labels = ["CIFAR-10 Resnet-18", "CIFAR-10 Conv-9", "tabular-benchmark", "blog-feedback", "titanic", "red-wine", "breast-cancer-wisconsin", "diabetes-readmission", "banking-marketing", "adult_income_dataset", "covertype", "higgs", "new-thyroid"]

group_descriptions = [
	"CIFAR-10: Resnet-18; lay=16, units<=65536 (ch*pix)\n                lin lay=1, units=512",
	"CIFAR-10: Conv-9; lay=6, units=12288 (ch*pix)\n                lin lay=2, units=1024",
	"tabular-benchmark: MLP-5; lay=3, units=64",
	"blog-feedback: MLP-5; lay=3, units=144",
	"titanic: MLP-5; lay=3, units=128",
	"red-wine: MLP-5; lay=3, units=128",
	"breast-cancer-wisconsin: MLP-5; lay=3, units=32",
	"diabetes-readmission: MLP-5; lay=3, units=304",
	"banking-marketing: MLP-6; lay=4, units=128",
	"adult_income_dataset: MLP-5; lay=3, units=256",
	"covertype: MLP-7; lay=5, units=512",
	"higgs: MLP-5; lay=3, units=256",
	"new-thyroid: MLP-4; lay=2, units=16",
]

# Convert to arrays
x = np.arange(len(subset(y1_values, graphedDatasets)))
y1 = np.array(subset(y1_values, graphedDatasets))
y2 = np.array(subset(y2_values, graphedDatasets))
width = 0.4

# Create figure and axes
fig, ax = plt.subplots(figsize=(12, 6))

# Plot bars
b1 = ax.bar(x, y1, width=width, color='blue', label='Full Backprop training')
b2 = ax.bar(x + width, y2, width=width, color='magenta', label='EISANI (useEIneurons=False)')

# Annotate bars
ax.bar_label(b1, fmt='%.1f', padding=3, fontsize=6)
ax.bar_label(b2, fmt='%.1f', padding=3, fontsize=6)

# X-axis group labels
ax.set_xticks(x + width/2)
ax.set_xticklabels(subset(group_labels, graphedDatasets), rotation=45, ha='right', fontsize=10)

# Minor ticks
#ax.minorticks_on()
#ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
ax.set_yticks(np.arange(0, 100+0.1, 10.0))

# Labels and title
ax.set_xlabel("dataset")
ax.set_ylabel("test accuracy (Top-1)")
ax.set_title("EISANI vs Full Backprop Training")

# Legend of series, moved to the right above the set descriptions
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 0.95), borderaxespad=0.)

# Adjust layout to make room for both legends and description key on the right
fig.subplots_adjust(right=0.8)

# Add descriptive key box on the right
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
key_text = "\n".join(subset(group_descriptions, graphedDatasets))
ax.text(1.02, 0.5, key_text, transform=ax.transAxes, fontsize=10, verticalalignment='center', bbox=props)

plt.tight_layout()
plt.show()
