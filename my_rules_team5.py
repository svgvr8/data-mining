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


def candidate_rules(freq_k_itemsets, min_confidence):
    candidate_rules = []
    #X -> Y - X
    for k in freq_k_itemsets:
        if k >= 2:
            for itemset in freq_k_itemsets[k]:
                #print(itemset)
                itemset = list(itemset)
                
                LHS =set(itemset[:-1])
                #print("LHS: ", LHS)
                RHS = set(itemset) - LHS
                #print("RHS: ", RHS)
                itemset = tuple(itemset)
                LHS = tuple(LHS)
                RHS = tuple(RHS)
                confidence = freq_k_itemsets[k][itemset]/ freq_k_itemsets[k-1][LHS] #get support from freq itemsets dictionary
                #print("confidence: ", confidence)
                #if confidence >= min_confidence:
                candidate_rules.append([LHS, RHS, confidence, freq_k_itemsets[k][itemset]])

    print("candidate rules: ", candidate_rules)

        
    return candidate_rules
     
def candidate_rules2(freq_k_itemsets, min_confidence):
    candidate_rules = []
    #X -> Y - X
    for k in freq_k_itemsets:
        if k >= 2:
            for itemset in freq_k_itemsets[k]:  
                itemset = list(itemset)                                 
                for RHS in itemset:
                    
                    LHS = set(itemset) - set(freq_k_itemsets[k][RHS])
                    print("LHS: ", LHS)
                    
                    LHS =set(itemset[:-1])
                    #print("LHS: ", LHS)
                    RHS = set(itemset) - LHS
                    #print("RHS: ", RHS)
                    itemset = tuple(itemset)
                    LHS = tuple(LHS)
                    RHS = tuple(RHS)
                    confidence = freq_k_itemsets[k][itemset]/ freq_k_itemsets[k-1][LHS] #get support from freq itemsets dictionary
                    #print("confidence: ", confidence)
                    #if confidence >= min_confidence:
                    candidate_rules.append([LHS, RHS, confidence, freq_k_itemsets[k][itemset]])

    print("candidate rules: ", candidate_rules)

        
    return candidate_rules
'''
    print("candidate rules: ", candidate_rules)

    for rule in candidate_rules:
            #LHS, RHS, confidence, support = rule
            #f.write
            #one rule per itemset
            #ex [1,2,3,4]

            for RHS in itemset:
                #LHS = set(itemset) - set(RHS)     
'''
def extract_k_counts(filename):
    k_counts = defaultdict(int)
    with open(filename, "r") as file:
        for line in file:
            items = line.split("|")[0].strip()[1:-1].split(", ")
            k = len(items)
            k_counts[k] += 1
    return k_counts

#----------------------START HERE----------------------
start = timeit
filename = f"{output_file_name}_items_team5.txt" #create file 1
print("Creating file 1: %s" % filename)
f = open(filename, "w")

F1 = gen_f1(transactions, minimum_support)
frequent_n_itemset = {} #set 1-itemsets for apriori beginning
frequent_n_itemset[1] = F1

k=2 #set k value to generate 2-itemsets

while(len(frequent_n_itemset[k-1]) > 1):
    frequent_n_itemset[k] = {}
    L = candidate_merge(frequent_n_itemset[k-1]) 
    print(f"{k}th level candidate itemsets: {len(L)}")

    L = pruning(L, frequent_n_itemset[k-1], k)
    print(f"{k}th level pruned itemsets: {len(L)}")

    frequent_n_itemset[k] = support_count(L, minimum_support, transactions, k)
    print(f"{k}th level frequent itemsets: {len(frequent_n_itemset[k])}")
    print(frequent_n_itemset[k])

    for freq_itemset, freq_support in frequent_n_itemset[k].items():
            f.write(f"{freq_itemset} | {freq_support}\n")

    s = candidate_rules2(frequent_n_itemset, minimum_confidence)
    print(s)
    k = k+1


f.close()
exit()
s = {}


for k in frequent_n_itemset:
    if k >= 2:
        for itemset in frequent_n_itemset[k]:
            print(itemset)
            s = candidate_rules(itemset, minimum_confidence)
      
print("s",s)            
    
exit()
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

k_counts = extract_k_counts(f"{output_file_name}_items_team5.txt")

# Extract data from the second file
k_counts_2 = extract_k_counts(f"{output_file_name}_items_team5_2.txt")

# Data setup
labels = sorted(set(k_counts.keys()).union(k_counts_2.keys()))
dataset1_counts = [k_counts[label] for label in labels]
dataset2_counts = [k_counts_2[label] for label in labels]

# Placeholder for the output file name
output_file_name = "<output_file_name>_plot_items_team5.png"

# Create the figure
plt.figure(figsize=(16, 10))

# Adjusted bar widths and positions for readability
bar_width = 0.4
index = np.arange(len(labels)) * 1.5  # increase spacing between bar groups

# Plotting bars
bar1 = plt.bar(
    index,
    dataset1_counts,
    bar_width,
    color="lightcoral",
    label="Dataset 1 (Support: 120, Confidence: X)",
)
bar2 = plt.bar(
    index + bar_width,
    dataset2_counts,
    bar_width,
    color="mediumseagreen",
    label="Dataset 2 (Support: 75, Confidence: X)",
)

# Increase font size for readability
plt.xlabel("k (Number of items in the itemset)", fontsize=14)
plt.ylabel("Number of frequent k-itemsets", fontsize=14)
plt.title("Comparison of frequent k-itemsets between two datasets", fontsize=18)
plt.xticks(index + bar_width / 2, labels, fontsize=14)  # position of tick labels
plt.yticks(fontsize=12)
plt.legend(fontsize=14)

# Add annotations on top of each bar with larger font size
for bars in [bar1, bar2]:
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 2,
            yval,
            ha="center",
            va="bottom",
            fontsize=14,
        )

# Enhance grid appearance
plt.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6)

# Save the plot to the specified file
plt.tight_layout()
plt.savefig(output_file_name)
plt.show()