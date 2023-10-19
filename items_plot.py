import matplotlib.pyplot as plt
from collections import defaultdict

# File paths for Dataset 1 and 2 (Using the same dataset for demonstration)
file_path1 = "min5008_items_1.txt"
file_path2 = "min5008_items_1.txt"

# Process Dataset 1
with open(file_path1, "r") as file:
    lines1 = file.readlines()

k_itemsets_count1 = defaultdict(int)
for line in lines1:
    itemset = line.split("|")[0].strip("{}")
    k = len(itemset.split(","))
    k_itemsets_count1[k] += 1
k_values1 = list(k_itemsets_count1.keys())
counts1 = list(k_itemsets_count1.values())

# Process Dataset 2
with open(file_path2, "r") as file:
    lines2 = file.readlines()

k_itemsets_count2 = defaultdict(int)
for line in lines2:
    itemset = line.split("|")[0].strip("{}")
    k = len(itemset.split(","))
    k_itemsets_count2[k] += 1
k_values2 = list(k_itemsets_count2.keys())
counts2 = list(k_itemsets_count2.values())

# Custom x-axis labels
custom_labels = ["1 = {x1}", "2 = {x1, x2}", "3 = {x1, x2, x3}", "4 = {x1, x2, x3, x4}"]
# Define the index for bar positions
index = range(len(k_values1))

# Enhanced plotting settings
width = 0.25  # narrower bars for better spacing

plt.figure(figsize=(14, 9))

# Bars for dataset 1
bars1 = plt.bar(
    index,
    counts1,
    width,
    color="royalblue",
    alpha=0.9,
    label="Dataset A: Support=50, Confidence=0.8",
)

# Bars for dataset 2 (offset by 'width' for side-by-side bars)
bars2 = plt.bar(
    [i + width for i in index],
    counts2,
    width,
    color="limegreen",
    alpha=0.9,
    label="Dataset B: Support=50, Confidence=0.8",
)

# Add annotations to bars
for bar in bars1:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 5,
        str(height),
        ha="center",
        va="bottom",
        fontsize=12,
    )

for bar in bars2:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 5,
        str(height),
        ha="center",
        va="bottom",
        fontsize=12,
    )

# Labels, title, and custom x-tick labels
plt.title("Number of Frequent k-itemsets for Different Values of k", fontsize=18)
plt.xlabel("k (Size and Example of Itemset)", fontsize=16, labelpad=15)
plt.ylabel("Number of Frequent Itemsets", fontsize=16, labelpad=15)
plt.xticks([i + width / 2 for i in index], custom_labels, fontsize=14)
plt.yticks(fontsize=14)

# Legend
plt.legend(fontsize=14)

# Save the enhanced plot
output_file_name = "output"
plt.savefig(f"{output_file_name}_plot_items_team5.png", dpi=300, bbox_inches="tight")

# Display the enhanced plot
plt.grid(axis="y")
plt.tight_layout()
plt.show()
