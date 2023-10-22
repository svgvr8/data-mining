import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

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

# Identify items that are commonly bought by customers
print("Identifying popular items...")

min_support_count = 120  # Set your desired threshold here
min_support = min_support_count / num_transactions  # Convert to relative support
min_conf = 0.8


def compute_frequent_itemsets(sparse_matrix, unique_items, min_support):
    frequent_itemsets = []
    for index, item in enumerate(unique_items):
        support = sparse_matrix[:, index].mean()
        if support >= min_support:
            frequent_itemsets.append((frozenset([item]), support))
    return frequent_itemsets


# Find combinations of items that are often bought together
print("Finding item combinations that are often bought together...")
frequent_itemsets = compute_frequent_itemsets(sparse_matrix, unique_items, min_support)
new_itemsets = frequent_itemsets

print(f"Identified {len(frequent_itemsets)} single items that are frequently bought.")

# Check bigger combinations (like pairs, triplets of items, etc.) to see which sets are popular
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
    print(
        f"Level {level}: Found {len(curr_level_itemsets)} popular combinations of items."
    )

# Save our findings into a file for future reference
output_file_name = "output_file_rules_team5.txt"
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

# Indicate the end of the process
print("Process completed.")
