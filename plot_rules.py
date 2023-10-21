import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
import matplotlib.pyplot as plt

# --- Read and organize transactions from the file ---
with open("small", "r") as file:
    lines = file.readlines()

transactions_dict = {}
for line in lines:
    transaction_id, item_id = map(int, line.split())
    if transaction_id not in transactions_dict:
        transactions_dict[transaction_id] = []
    transactions_dict[transaction_id].append(item_id)

transactions = list(transactions_dict.values())

# --- Begin the data processing ---
print("Starting the process...")

# Organizing each sale into a sparse matrix for memory efficiency
print("Organizing sales into a table...")
unique_items = sorted(list(set(item for sublist in transactions for item in sublist)))
num_transactions = len(transactions)
num_items = len(unique_items)

sparse_matrix = lil_matrix((num_transactions, num_items), dtype=int)
for i, transaction in enumerate(transactions):
    indices = [unique_items.index(item) for item in transaction]
    sparse_matrix[i, indices] = 1

print(f"Sparse matrix created with shape: {sparse_matrix.shape}")

# Convert the matrix to CSR format for efficient row-based operations
sparse_matrix = csr_matrix(sparse_matrix)


def compute_frequent_itemsets(sparse_matrix, unique_items, min_support):
    frequent_itemsets = []
    for index, item in enumerate(unique_items):
        support = sparse_matrix[:, index].mean()
        if support >= min_support:
            frequent_itemsets.append((frozenset([item]), support))
    return frequent_itemsets


def plot_frequent_itemsets(min_support_count, min_conf, color):
    min_support = min_support_count / num_transactions
    frequent_itemsets = compute_frequent_itemsets(
        sparse_matrix, unique_items, min_support
    )
    new_itemsets = frequent_itemsets
    k_counts = {}

    level = 1
    while new_itemsets:
        curr_level_itemsets = []
        for itemset1, support1 in new_itemsets:
            for itemset2, support2 in frequent_itemsets:
                merged = itemset1.union(itemset2)
                if len(merged) == len(itemset1) + 1:
                    indices = [unique_items.index(item) for item in merged]
                    merged_support = (
                        sparse_matrix[:, indices].sum(axis=1).A1 == len(merged)
                    ).mean()
                    if merged_support >= min_support:
                        if (merged, merged_support) not in curr_level_itemsets:
                            curr_level_itemsets.append((merged, merged_support))
        k_counts[level] = len(curr_level_itemsets)
        frequent_itemsets.extend(curr_level_itemsets)
        new_itemsets = curr_level_itemsets
        level += 1

    output_file_name = (
        f"output_file_rules_support_{min_support_count}_conf_{min_conf}.txt"
    )
    if min_conf != -1:
        with open(output_file_name, "w") as file:
            for itemset, support in frequent_itemsets:
                for item in itemset:
                    antecedent = itemset - frozenset([item])
                    consequent = frozenset([item])
                    if antecedent:
                        antecedent_indices = [
                            unique_items.index(item) for item in antecedent
                        ]
                        antecedent_support = (
                            sparse_matrix[:, antecedent_indices].sum(axis=1).A1
                            == len(antecedent)
                        ).mean()
                        confidence = support / antecedent_support
                        if confidence >= min_conf:
                            file.write(
                                f"{' '.join(map(str, antecedent))}|{' '.join(map(str, consequent))}|{support:.2f}|{confidence:.2f}\n"
                            )
        print(f"Saved our findings in {output_file_name}")

    return k_counts


k_counts_1 = plot_frequent_itemsets(100, 0.1, "skyblue")
k_counts_2 = plot_frequent_itemsets(200, 0.2, "salmon")

bar_width = 0.35
k_values = sorted(set(list(k_counts_1.keys()) + list(k_counts_2.keys())))
counts_1 = [k_counts_1.get(k, 0) for k in k_values]
counts_2 = [k_counts_2.get(k, 0) for k in k_values]


plt.figure(figsize=(15, 7))

# Adjust bar width for better readability
bar_width = 0.4

# Adjust positions for bars
positions_1 = [i - bar_width / 2 for i in k_values]
positions_2 = [i + bar_width / 2 for i in k_values]

rects1 = plt.bar(
    positions_1,
    counts_1,
    bar_width,
    color="skyblue",
    label="Support: 100, Conf: 0.1",
    edgecolor="black",
)
rects2 = plt.bar(
    positions_2,
    counts_2,
    bar_width,
    color="salmon",
    label="Support: 200, Conf: 0.2",
    edgecolor="black",
)


# Enhance annotations above each bar
def autolabel(rects, xpos="center"):
    ha = {"center": "center", "right": "left", "left": "right"}
    offset = {"center": 0, "right": 1, "left": -1}

    for rect in rects:
        height = rect.get_height()
        plt.text(
            rect.get_x() + rect.get_width() * offset[xpos],
            1.05 * height,
            f"{height}",
            ha=ha[xpos],
            va="bottom",
            fontweight="bold",
        )


autolabel(rects1, "center")
autolabel(rects2, "center")

plt.xlabel("Number of Antecedent Items (k)", fontsize=14)
plt.ylabel("Number of High-Confidence Rules", fontsize=14)
plt.title(
    "Comparison of High-Confidence Rules for Different Values of Support and Confidence",
    fontsize=16,
)
plt.xticks(k_values, fontsize=13)
plt.yticks(fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Display legend
plt.legend(fontsize=12)

# Tight layout for saving the plot without cutting off
plt.tight_layout()

output_file_name = "comparison_plot_rules_team5.png"
plt.savefig(output_file_name)
plt.show()
