#!/usr/bin/env python
# coding: utf-8

# # Project 1 Codes (Team # 5)

# ## Function to Load Data (Loading Transactions from the File)

# #### This function is receiving the data path from its location and read/load it lexicografically (process of sorting the list of items in the data in lexicographic order).


import pandas as pd
import sys

minimum_support = int(sys.argv[1])
minimum_confidence = int(sys.argv[2])
input_file_name = sys.argv[3]
output_file_name = sys.argv[4]


def read_transactions(
    input_file,
):  # Defining the (read_transactions) that takes the parameter “path to the file containing transaction data into “, input_file”.
    transactions = []  # Creating a list for appending (storing) the transactions
    with open(
        input_file, "r"
    ) as file:  # Opening the file where the transactions are (data, 'r' here means read)
        for line in file:  # Looping over each line in the file to read
            transaction = (
                line.strip().split()
            )  # Removes leading and trailing whitespace characters for each line in the file using strip() and then splits the line into a list of strings using the default whitespace separator (spaces and tabs). This assumes that each line in the file contains space-separated values, and each value represents a part of a transaction
            transactions.append(
                transaction
            )  # Once splitting the line into a list of strings, the resulting list (representing a transaction) is added (stored) to the transactions list. After the loop finishes, the with block ensures that the file is properly closed.
    return transactions  # Finally, after building up the list in organized fashion (read the data from the file and organize it into a list of transactions), the function returns the transactions list.


# input_file = "C:/Users/sebi2/OneDrive/Desktop/School/Fall 2023/datamining/project/small" # Definingn the full path to the data file using
input_file = input_file_name

transactions = read_transactions(input_file)  # reading the file

# Now, 'transactions' contains a list of transactions from the file


print(transactions)


# ## Generating F1 list (List of 1-itemset)


# Initialize variables to store minimum and maximum values
min_value = float("inf")  # Initialize to positive infinity
max_value = float("-inf")  # Initialize to negative infinity

# Iterate through the dataset and update min and max values
for prefix, value in transactions:
    # Convert 'value' to a float before comparison
    value = float(value)

    if value < min_value:
        min_value = value
    if value > max_value:
        max_value = value

# Print the minimum and maximum values
print(f"Minimum value: {min_value}")
print(f"Maximum value: {max_value}")


from collections import Counter


def generate_F1(transactions, min_support):
    item_counts = Counter(item for transaction in transactions for item in transaction)

    F1 = {
        item: support for item, support in item_counts.items() if support >= min_support
    }

    # Sort the F1 itemsets in descending order of support count
    F1_sorted = dict(sorted(F1.items(), key=lambda item: item[1], reverse=True))

    return F1_sorted


# =============================================================
# output file 1
min_support = minimum_support

F1 = generate_F1(transactions, min_support)

print("Frequent 1-Itemsets (F1) in Descending Order of Support:")
for item, support in F1.items():
    print(f"{item}: {support}")


filename = f"{output_file_name}_items_team5.txt"

print("Creating file 1: %s" % filename)
f = open(filename, "w")

for item, support in F1.items():
    f.write(f"{item} | {support}\n")

f.close()
# ======================================================
# output file 2
if minimum_confidence != -1:
    filename = f"{output_file_name}_rules_team5.txt"

    print("Creating file 2: %s" % filename)
    f = open(filename, "w")

    for item, support in F1.items():
        f.write(f"LHS | RHS | SUPPORT | CONFIDENCE\n")

    f.close()
else:
    print("confidence = -1, not generating rules file")
# ======================================================
# output file 3
filename = f"{output_file_name}_info_team5.txt"

print("Creating file 3: %s" % filename)
f = open(filename, "w")


f.write(
    f"""minsup: 
            minconf: 
            input file: 
            output name: 
            Number of items:
            Number of transactions: 
            Length of largest frequent k-itemset: 
            Number of frequent 1-itemsets: 
            ...
            Number of frequent k-itemsets: 
            Total number of frequent itemsets: 
            Number of high confidence rules: 
            Rule with highest confidence: 
            Time to fine the frequent itemsets (s): 
            Time to find confident rules (s): """
)

f.close()
