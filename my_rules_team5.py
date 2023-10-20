#CAP 5771 Team 5
import pandas as pd
import sys
import matplotlib.pyplot as plt
from collections import Counter
import itertools
from collections import defaultdict
import timeit
import os

#set arguments
minimum_support = int(sys.argv[1]) 
minimum_confidence = float(sys.argv[2])
input_file_name = sys.argv[3]
output_file_name = sys.argv[4]

input_file = input_file_name


def read_data(file_name): # Function to read data from a file and return a list of transactions
    transactions = defaultdict(list)

    with open(file_name, 'r') as file:
        for line in file:
            transaction_id, item_id = map(int, line.strip().split())
            transactions[transaction_id].append(item_id)

    return transactions

transactions = read_data(input_file)  #read input file

#print(transactions)

def gen_f1(transactions, min_support): # frequent 1-itemset generation function, needed to start Apriori algorithm
    item_counts = defaultdict(int)

    # Count the occurrences of each item in the transactions
    for transaction in transactions.values():  # Iterate over the lists of item IDs
        for item in transaction:
            item =  tuple([item])
            item_counts[item] += 1

    frequent_1_itemsets = {item: support for item, support in item_counts.items() if support >= min_support} # Filter items based on minimum support
    #F1_sorted = {item: support for item, support in sorted(frequent_1_itemsets.items(), key=lambda x: x[1], reverse=True)} #order items in descending support count
    return frequent_1_itemsets

                                                                          #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



def candidate_merge(prev_freq_itemsets):
    candidate_itemsets = []
      
#For k=1
    temp_k = len(list(prev_freq_itemsets.keys())[1]) #if k=1

    for i in range(len(prev_freq_itemsets)):#transaction
        for j in range(i + 1, len(prev_freq_itemsets)): #items
            
            itemset1 = list(prev_freq_itemsets.keys())[i]
            itemset2 = list(prev_freq_itemsets.keys())[j]
            if temp_k == 1:
                candidate_itemsets.append(tuple(sorted(set(itemset1 + itemset2))))
            
            elif itemset1[:-1] == itemset2[:-1]:
                # Merge (k-1)-itemsets to create a candidate k-itemset
                candidate_itemsets.append(tuple(sorted(set(itemset1 + itemset2))))    
    
    
    return candidate_itemsets

def pruning(candidate_itemset, k_minus_1_itemset, k): 
    pruned_itemsets = []
    
    if k>2:
        for candidate in candidate_itemset:
            #print("candidate: ", candidate)
            subsets = list(itertools.combinations(candidate,k-1))
            #print("subsets: ",*subsets)
            for subset in subsets:
                #print("subset: ",subset)
                #print("k-1: ", k_minus_1_itemset)

                if subset in k_minus_1_itemset.keys(): #k-1 is a dictionary
                    pruned_itemsets.append(candidate) 
                    break    #exit loop if first subset is in 

        #print("pruned itemset: ",pruned_itemsets)
        return pruned_itemsets #list of lists
    
    else:
        return candidate_itemset

def support_count(candidate_itemsets,min_support, transactions, k):
    support_itemsets = {}
    removed_itemsets = []

    for candidate in candidate_itemsets:
        support_itemsets[candidate] = 0

    for transaction_id in transactions:
        for k_itemset in list(itertools.combinations(transactions[transaction_id], k)):
            if k_itemset in support_itemsets:
                support_itemsets[k_itemset] += 1

    for itemset in support_itemsets:
        if support_itemsets[itemset] < min_support:
            removed_itemsets.append(itemset)

    for itemset in removed_itemsets:
        support_itemsets.pop(itemset, None)

    return support_itemsets

def confidence(frequent_n_itemset, min_confidence, k):
    
    confidence_rules = []
    for k in frequent_n_itemset: 
        for itemset in frequent_n_itemset[k]:

            LHS = {itemset}
            RHS = set(itemset) - LHS
            confidence = LHS.values() / RHS.values()

            if confidence >= min_confidence:
                confidence_rules.append((LHS, RHS, confidence))

            #generate candidates
            #start with rules that have 1 items on RHS

            #create dictionary and store set as key
            #define keys

            #Rules = {rhs:conf}
            #list of sets
    
    return confidence_rules

#----------------------START HERE----------------------

filename = f"{output_file_name}_items_team5.txt" #create file 1

print("Creating file 1: %s" % filename)

f = open(filename, "w")

F1 = gen_f1(transactions, minimum_support)
frequent_n_itemset = {} #set 1-itemsets for apriori beginning
frequent_n_itemset[1] = F1

k=2 #set k value to generate 2-itemsets

while(len(frequent_n_itemset[k-1]) > 1):
    frequent_n_itemset[k] = {}
    L = candidate_merge(frequent_n_itemset[k-1]) #pruning input #if subset is infrequent, remove from L
    print(f"{k}th level candidate itemsets: {len(L)}")
    L = pruning(L, frequent_n_itemset[k-1], k)
    print(f"{k}th level pruned itemsets: {len(L)}")
    frequent_n_itemset[k] = support_count(L, minimum_support, transactions, k)
    print(f"{k}th level frequent itemsets: {len(frequent_n_itemset[k])}")
    print(frequent_n_itemset[k])


    for freq_itemset, freq_support in frequent_n_itemset[k].items():
            f.write(f"{freq_itemset} | {freq_support}\n")

    k = k+1

f.close()
exit()
    #pruning, only append itemsets that survived
    #support count
    #elimination



#rule generation



#create output file        




# ======================================================
# output file 2                                                 Will be completed in Phase 3
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
# output file 3                                             Will be completed in Phase 3
filename = f"{output_file_name}_info_team5.txt"

print("Creating file 3: %s" % filename)
f = open(filename, "w")

f.write(
    f"""
            minsup: {minimum_support} 
            minconf: {minimum_confidence}
            input file: {input_file_name}
            output name: {output_file_name}
            Number of items: {transactions.values()}
            Number of transactions: {len(transactions)}
            Length of largest frequent k-itemset: Placeholder\n"""
)
f.write(
    f"""            
            Number of frequent 1-itemsets: {len(F1.items)}
            ...
            Number of frequent k-itemsets: 
            Total number of frequent itemsets: 
            Number of high confidence rules: 
            Rule with highest confidence: 
            Time to find the frequent itemsets (s): {stop} - {start}
            Time to find confident rules (s): """
)            

f.close()
#======================================
#generate itemset plot

# Initialize empty dictionary to store the support counts
support_counts = {}

# Read data from the file and populate the support_counts dictionary
with open(f"{output_file_name}_items_team5.txt", "r") as file:
    for line in file:
        item, count = line.strip().split(" | ")
        support_counts[int(item)] = int(count)

# Sort the support counts in increasing order by their values
sorted_support_counts = sorted(support_counts.items(), key=lambda x: x[1])

# Extract items and support counts after sorting
items, counts = zip(*sorted_support_counts)

# Create subplots for each group of 20 items
num_subplots = (len(items) // 50) + 1

for i in range(num_subplots):
    start_index = i * 50
    end_index = min((i + 1) * 50, len(items))

    # Create a bar plot for the current group of 20 items
    plt.figure(figsize=(10, 6))
    plt.bar(
        range(end_index - start_index),
        counts[start_index:end_index],
        tick_label=items[start_index:end_index],
    )
    plt.xlabel("Items")
    plt.ylabel("Support Count")
    plt.title(
        f"Support Counts in Increasing Order (Items {start_index + 1}-{end_index})"
    )

    # Rotate x-axis labels for better readability (optional)
    plt.xticks(rotation=45)

    # Show the plot
    plt.tight_layout()
    plt.show()

plt.savefig("plot.png")