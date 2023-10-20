from itertools import combinations

# Reading the data
with open("small", "r") as file:
    transactions = [line.strip().split() for line in file]

# Hard coded values
minsup = 0.016361256544502618
minconf = 0.8


# Apriori function to compute frequent itemsets
def apriori(transactions, minsup):
    all_items = set(item for transaction in transactions for item in transaction)
    candidates = [frozenset([item]) for item in all_items]
    frequent_itemsets = []

    while candidates:
        itemset_counts = {candidate: 0 for candidate in candidates}

        for transaction in transactions:
            for candidate in candidates:
                if candidate.issubset(transaction):
                    itemset_counts[candidate] += 1

        new_frequent_itemsets = [
            itemset
            for itemset, count in itemset_counts.items()
            if count / len(transactions) >= minsup
        ]
        frequent_itemsets.extend(new_frequent_itemsets)

        candidates = set()
        for itemset1 in new_frequent_itemsets:
            for itemset2 in new_frequent_itemsets:
                new_itemset = itemset1.union(itemset2)
                if len(new_itemset) == len(itemset1) + 1:
                    candidates.add(frozenset(new_itemset))
        candidates = list(candidates)

    return frequent_itemsets


# Function to generate high confidence rules
def generate_rules(frequent_itemsets, minconf):
    rules = []
    for itemset in frequent_itemsets:
        n = len(itemset)
        if n > 1:
            for i in range(1, n):
                for antecedent in combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    antecedent_support = sum(
                        1
                        for transaction in transactions
                        if antecedent.issubset(transaction)
                    ) / len(transactions)
                    itemset_support = sum(
                        1
                        for transaction in transactions
                        if itemset.issubset(transaction)
                    ) / len(transactions)
                    confidence = itemset_support / antecedent_support
                    if confidence >= minconf:
                        rules.append((antecedent, consequent, confidence))
    return rules


# Compute the frequent itemsets and high confidence rules
frequent_itemsets = apriori(transactions, minsup)
confident_rules = generate_rules(frequent_itemsets, minconf)

# Extract metrics
output = {
    "Number of items": len(
        set(item for transaction in transactions for item in transaction)
    ),
    "Number of transactions": len(transactions),
    "The length of the largest frequent k-itemset": max(map(len, frequent_itemsets)),
    "Number of frequent 1-itemsets": sum(
        1 for itemset in frequent_itemsets if len(itemset) == 1
    ),
    "Number of frequent 2-itemsets": sum(
        1 for itemset in frequent_itemsets if len(itemset) == 2
    ),
    "Number of frequent 3-itemsets": sum(
        1 for itemset in frequent_itemsets if len(itemset) == 3
    ),
    "Number of frequent 4-itemsets": sum(
        1 for itemset in frequent_itemsets if len(itemset) == 4
    ),
    "Total number of frequent itemsets": len(frequent_itemsets),
    "Number of high confidence rules": len(confident_rules),
    "The rule with the highest confidence": max(
        confident_rules, key=lambda rule: rule[2]
    )
    if confident_rules
    else None,
}

print(output)
