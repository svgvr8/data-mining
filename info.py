import pandas as pd
import numpy as np
from itertools import combinations
import time

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

# Convert to DataFrame
df = pd.DataFrame(data, columns=["item_1", "item_2"])
df = pd.get_dummies(df.stack()).groupby(level=0).sum()

minsup = 0.010
mincof = 0.8


# Function to compute frequent itemsets
def apriori_frequent_itemsets(df, minsup):
    num_transactions = len(df)
    items = df.columns.tolist()
    frequent_itemsets = []

    k = 1
    candidates = [[item] for item in items]

    while candidates:
        valid_candidates = []
        for candidate in candidates:
            support = df[candidate].all(axis=1).mean()
            if support >= minsup:
                frequent_itemsets.append((frozenset(candidate), support))
                valid_candidates.append(candidate)

        next_candidates = []
        for i in range(len(valid_candidates)):
            for j in range(i + 1, len(valid_candidates)):
                merged = list(set(valid_candidates[i]) | set(valid_candidates[j]))
                if len(merged) == k + 1:
                    next_candidates.append(merged)

        candidates = next_candidates
        k += 1

    return frequent_itemsets


# Derive association rules
def derive_association_rules(frequent_itemsets, mincof):
    rules = []
    for itemset, support in frequent_itemsets:
        k = len(itemset)
        if k > 1:
            for i in range(1, k):
                antecedents = [frozenset(comb) for comb in combinations(itemset, i)]
                for antecedent in antecedents:
                    consequent = itemset - antecedent
                    antecedent_support = [
                        s
                        for s_itemset, s in frequent_itemsets
                        if s_itemset == antecedent
                    ][0]
                    confidence = support / antecedent_support
                    if confidence >= mincof:
                        rules.append((antecedent, consequent, support, confidence))
    return rules


start_time = time.time()
frequent_itemsets = apriori_frequent_itemsets(df, minsup)
end_time = time.time()
time_frequent_itemsets = end_time - start_time

start_time = time.time()
association_rules = derive_association_rules(frequent_itemsets, mincof)
end_time = time.time()
time_confident_rules = end_time - start_time

num_items = df.shape[1]
num_transactions = df.shape[0]
max_k = max([len(itemset) for itemset, _ in frequent_itemsets])
k_counts = (
    pd.Series([len(itemset) for itemset, _ in frequent_itemsets])
    .value_counts()
    .sort_index()
)
total_frequent_itemsets = len(frequent_itemsets)
num_high_confidence_rules = len(association_rules)
highest_conf_rule = max(association_rules, key=lambda x: x[3])

info = {
    "Number of items": num_items,
    "Number of transactions": num_transactions,
    "Largest frequent k-itemset length": max_k,
    "Total number of frequent itemsets": total_frequent_itemsets,
    "Number of high confidence rules": num_high_confidence_rules,
    "Rule with the highest confidence": highest_conf_rule,
    "Time to find frequent itemsets (seconds)": time_frequent_itemsets,
    "Time to find confident rules (seconds)": time_confident_rules,
}

for k, count in k_counts.items():
    info[f"Number of frequent {k}-itemsets"] = count

for key, value in info.items():
    print(f"{key}: {value}")
