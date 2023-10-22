import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


# Function to process the file and extract k-itemset counts
def extract_k_counts(filename):
    k_counts = defaultdict(int)
    with open(filename, "r") as file:
        for line in file:
            items = line.split("|")[0].strip()[1:-1].split(", ")
            k = len(items)
            k_counts[k] += 1
    return k_counts


# Extract data from the first file
k_counts = extract_k_counts("test_items_team5.txt")

# Extract data from the second file
k_counts_2 = extract_k_counts("test_items_team5_2.txt")

# Data setup
labels = sorted(set(k_counts.keys()).union(k_counts_2.keys()))
dataset1_counts = [k_counts[label] for label in labels]
dataset2_counts = [k_counts_2[label] for label in labels]

# Placeholder for the output file name
output_file_name = "test_plot_items_team5.png"

# Create the figure
plt.figure(figsize=(16, 10))

# Adjusted bar widths and positions for readability
bar_width = 0.4
index = np.arange(len(labels)) * 1.5  # increase spacing between bar groups

# Plotting bars
bar1 = plt.bar(
    index,
    dataset1_counts,
    bar_width,
    color="lightcoral",
    label="Dataset 1 (Support: 120, Confidence: X)",
)
bar2 = plt.bar(
    index + bar_width,
    dataset2_counts,
    bar_width,
    color="mediumseagreen",
    label="Dataset 2 (Support: 75, Confidence: X)",
)

# Increase font size for readability
plt.xlabel("k (Number of items in the itemset)", fontsize=14)
plt.ylabel("Number of frequent k-itemsets", fontsize=14)
plt.title("Comparison of frequent k-itemsets between two datasets", fontsize=18)
plt.xticks(index + bar_width / 2, labels, fontsize=14)  # position of tick labels
plt.yticks(fontsize=12)
plt.legend(fontsize=14)

# Add annotations on top of each bar with larger font size
for bars in [bar1, bar2]:
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 2,
            yval,
            ha="center",
            va="bottom",
            fontsize=14,
        )

# Enhance grid appearance
plt.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6)

# Save the plot to the specified file
plt.tight_layout()
plt.savefig(output_file_name)
plt.show()
