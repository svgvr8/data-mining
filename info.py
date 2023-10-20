from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd
import time

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

te = TransactionEncoder()
te_ary = te.fit(data).transform(data)
df = pd.DataFrame(te_ary, columns=te.columns_)


minsup = 0.010
frequent_itemsets = apriori(df, min_support=minsup, use_colnames=True)

mincof = 0.8
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=mincof)

# Number of items
num_items = len(te.columns_)

# Number of transactions
num_transactions = len(data)

# Length of the largest frequent k-itemset
max_k = frequent_itemsets["itemsets"].apply(len).max()

# Number of frequent k-itemsets
k_counts = frequent_itemsets["itemsets"].apply(len).value_counts().sort_index()

# Total number of frequent itemsets
total_frequent_itemsets = len(frequent_itemsets)

# Number of high confidence rules
num_high_confidence_rules = len(rules)

# The rule with the highest confidence
highest_conf_rule = rules.iloc[rules["confidence"].idxmax()]

# Print Results
print(f"Number of items: {num_items}")
print(f"Number of transactions: {num_transactions}")
print(f"The length of the largest frequent k-itemset: {max_k}")
for k, count in k_counts.items():
    print(f"Number of frequent {k}-itemsets: {count}")
print(f"Total number of frequent itemsets: {total_frequent_itemsets}")
print(f"Number of high confidence rules: {num_high_confidence_rules}")
print(
    f"The rule with the highest confidence: {highest_conf_rule['antecedents']} => {highest_conf_rule['consequents']} with confidence {highest_conf_rule['confidence']}"
)


start_time = time.time()
frequent_itemsets = apriori(df, min_support=minsup, use_colnames=True)
end_time = time.time()
print(f"Time in seconds to find the frequent itemsets: {end_time - start_time}")

start_time = time.time()
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=mincof)
end_time = time.time()
print(f"Time in seconds to find the confident rules: {end_time - start_time}")
