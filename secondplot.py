import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
import matplotlib.pyplot as plt
import time
import matplotlib

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

# Organizing each sale into a sparse matrix for memory efficiency
unique_items = sorted(list(set(item for sublist in transactions for item in sublist)))
num_transactions = len(transactions)
num_items = len(unique_items)

sparse_matrix = lil_matrix((num_transactions, num_items), dtype=int)
for i, transaction in enumerate(transactions):
    indices = [unique_items.index(item) for item in transaction]
    sparse_matrix[i, indices] = 1

# Convert the matrix to CSR format for efficient row-based operations
sparse_matrix = csr_matrix(sparse_matrix)


def compute_frequent_itemsets(sparse_matrix, unique_items, min_support):
    frequent_itemsets = []
    for index, item in enumerate(unique_items):
        support = sparse_matrix[:, index].mean()
        if support >= min_support:
            frequent_itemsets.append((frozenset([item]), support))
    return frequent_itemsets


# Define the list of min_support_count and min_conf values
min_support_counts = [50, 100, 150, 200]
min_confs = [0.8, 0.95]

# Store the counts of itemsets and rules
counts = {conf: {"itemsets": [], "rules": []} for conf in min_confs}

for min_conf in min_confs:
    for min_support_count in min_support_counts:
        min_support = min_support_count / num_transactions
        frequent_itemsets = compute_frequent_itemsets(
            sparse_matrix, unique_items, min_support
        )
        new_itemsets = frequent_itemsets
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
            frequent_itemsets.extend(curr_level_itemsets)
            new_itemsets = curr_level_itemsets
            level += 1

        counts[min_conf]["itemsets"].append(len(frequent_itemsets))

        # Counting the rules
        rule_count = 0
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
                        rule_count += 1
        counts[min_conf]["rules"].append(rule_count)

# Now, plot and save the results
for min_conf in min_confs:
    plt.figure(figsize=(12, 7))
    bars1 = plt.bar(
        np.array(range(len(min_support_counts))) - 0.2,
        counts[min_conf]["itemsets"],
        0.4,
        label="Frequent Itemsets",
        color="blue",
        edgecolor="black",
    )
    bars2 = plt.bar(
        np.array(range(len(min_support_counts))) + 0.2,
        counts[min_conf]["rules"],
        0.4,
        label="Rules",
        color="green",
        edgecolor="black",
    )

    # Add value labels on top of the bars
    for bar in bars1:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.5,
            str(yval),
            ha="center",
            va="bottom",
        )
    for bar in bars2:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.5,
            str(yval),
            ha="center",
            va="bottom",
        )

    plt.xticks(range(len(min_support_counts)), min_support_counts)
    plt.xlabel("Minimum Support Count", fontsize=16)
    plt.ylabel("Count", fontsize=16)
    plt.title(f"Number of Itemsets and Rules for min_conf = {min_conf}", fontsize=18)
    plt.legend(loc="upper right", fontsize=14)

    # Save the plot to a file
    file_name = f"SECONDmin_conf_{min_conf}.png"
    plt.tight_layout()
    plt.savefig(file_name)
    print(f"Plot saved as {file_name}")

    plt.close()
