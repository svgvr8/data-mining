import pandas as pd
import numpy as np
from itertools import combinations


# Read transactions from "small.txt"
def read_transactions(file_name):
    with open(file_name, "r") as f:
        transactions = [list(map(int, line.strip().split())) for line in f.readlines()]
    return transactions


# Convert data to one-hot encoded DataFrame
def encode_data(data):
    df = pd.DataFrame(data)
    encoded_df = pd.get_dummies(df.stack()).groupby(level=0).sum()
    return encoded_df


# data = read_transactions("small")

# Provided data
data = [
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 4],
    [0, 5],
    [0, 6],
    [0, 7],
    [0, 8],
    [0, 9],
    [0, 10],
    [0, 11],
    [0, 12],
    [0, 13],
    [0, 14],
    [0, 15],
    [0, 16],
]

# Generate rules for specified confidence thresholds and save them to files
minconfs = [0.8, 0.95]


# Generate frequent itemsets
def apriori(data, min_support=0.01):
    df = encode_data(data)
    frequent_itemsets = {}
    n_rows = df.shape[0]

    # Get single itemsets
    single_itemsets = df.columns[df.sum(axis=0) / n_rows >= min_support].tolist()
    k = 1
    current_itemsets = single_itemsets

    while current_itemsets:
        frequent_itemsets[k] = current_itemsets
        k += 1
        current_itemsets = [
            tuple(sorted(itemset)) for itemset in combinations(single_itemsets, k)
        ]
        current_itemsets = [
            itemset
            for itemset in current_itemsets
            if df[list(itemset)].all(axis=1).sum() / n_rows >= min_support
        ]

    return frequent_itemsets


# Generate association rules
def generate_rules(data, min_support=0.01, min_confidence=0.01):
    df = encode_data(data)
    n_rows = df.shape[0]
    frequent_itemsets = apriori(data, min_support)
    rules = []

    for k, itemsets in frequent_itemsets.items():
        if k == 1:
            continue
        for itemset in itemsets:
            for i in range(1, k):
                lefts = list(combinations(itemset, i))
                for left in lefts:
                    right = tuple(sorted(set(itemset) - set(left)))
                    left_support = df[list(left)].all(axis=1).sum() / n_rows
                    both_support = df[list(itemset)].all(axis=1).sum() / n_rows
                    confidence = both_support / left_support
                    if confidence >= min_confidence:
                        rules.append((left, right, both_support, confidence))

    return rules


for minconf in minconfs:
    rules = generate_rules(data, min_support=0.01, min_confidence=minconf)
    file_name = f"output_file_rules_{minconf}.txt"
    with open(file_name, "w") as f:
        for rule in rules:
            f.write(
                f"{' '.join(map(str, rule[0]))}|{' '.join(map(str, rule[1]))}|{rule[2]:.4f}|{rule[3]:.4f}\n"
            )
