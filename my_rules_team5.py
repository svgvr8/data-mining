#CAP 5771 Team 5
import pandas as pd
import sys
import matplotlib.pyplot as plt
from collections import Counter
import itertools
from collections import defaultdict
import timeit
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
import time

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

    return transactions #transactions are stored as a dictionary with keys=transaction, values=itemsets

transactions = read_data(input_file)  #read input file

def gen_f1(transactions, min_support): # Function to generate frequent 1-itemsets, needed to start Apriori algorithm
    item_counts = defaultdict(int)

    # Count the occurrences of each item in the transactions
    for transaction in transactions.values():  # Iterate over the lists of item IDs
        for item in transaction:
            item =  tuple([item])
            item_counts[item] += 1

    frequent_1_itemsets = {item: support for item, support in item_counts.items() if support >= min_support} #Filter items based on minimum support
    return frequent_1_itemsets

def candidate_merge(prev_freq_itemsets): # Function to generate candidate L-itemsets
    candidate_itemsets = [] #storing candidate itemsets as a list of lists
      
    temp_k = len(list(prev_freq_itemsets.keys())[1]) #if k=1

    for i in range(len(prev_freq_itemsets)):#transaction
        for j in range(i + 1, len(prev_freq_itemsets)): #items
            
            itemset1 = list(prev_freq_itemsets.keys())[i] #first itemset
            itemset2 = list(prev_freq_itemsets.keys())[j] #second itemset
            if temp_k == 1:
                candidate_itemsets.append(tuple(sorted(set(itemset1 + itemset2)))) #merges first and second itemset to produce L+1 candidate itemset
            
            elif itemset1[:-1] == itemset2[:-1]:
                candidate_itemsets.append(tuple(sorted(set(itemset1 + itemset2)))) # Merge (k-1)-itemsets to create a candidate k-itemset    
    
    return candidate_itemsets

def pruning(candidate_itemset, k_minus_1_itemset, k): # Function to prune candidate itemsets that are not frequent
    pruned_itemsets = [] #storing pruned itemsets as a list of lists
    if k>=2: #can only prune itemsets are longer than 1 item
        for candidate in candidate_itemset:
            subsets = list(itertools.combinations(candidate,k-1)) #generates subsets used to check if candidate itemset it frequent
    
            for subset in subsets:
                if subset in k_minus_1_itemset.keys(): #k-1 is a dictionary
                    pruned_itemsets.append(candidate) 
                    break    #exit loop if first subset is in itemset
        
        return pruned_itemsets #list of lists
    
    else:
        return candidate_itemset

def support_count(candidate_itemsets,min_support, transactions, k): # Function to count the support of each itemset
    support_itemsets = {} #storing frequent itemsets
    removed_itemsets = [] #infrequent itemsets that will be removed

    for candidate in candidate_itemsets:
        support_itemsets[candidate] = 0 #initialize support count for each itemset

    for transaction_id in transactions:
        for k_itemset in list(itertools.combinations(transactions[transaction_id], k)):
            if k_itemset in support_itemsets:
                support_itemsets[k_itemset] += 1 #count how many times an itemset appears

    for itemset in support_itemsets: 
        if support_itemsets[itemset] < min_support:
            removed_itemsets.append(itemset) #add infrequent itemsets to list for removal

    for itemset in removed_itemsets:
        support_itemsets.pop(itemset, None) #remove infrequent itemsets

    return support_itemsets


def candidate_rules(freq_k_itemsets, min_confidence): # Function to generate association rules
    confidence_rules = [] #Storing rules
    
    for k in freq_k_itemsets:
        if k >= 2: #can only generate rules if itemset length is 2 or greater
            for itemset in freq_k_itemsets[k]:
                for RHS in itemset:
                    LHS = set(itemset) - set([RHS])  #set LHS

                    LHS = tuple(LHS)
                    if len(LHS) >= 1: #calculate support and confidence
                        confidence = freq_k_itemsets[k][itemset]/ freq_k_itemsets[k-1][LHS] #get support count from freq itemsets dictionary
                        support = freq_k_itemsets[k][itemset]/len(transactions)
                        print("confidence: ", confidence)
                        
                        if confidence >= min_confidence:
                            confidence_rules.append([LHS, RHS,round(support,2), round(confidence,2)]) #add rules that pass min confidence
    
    print("confidence rules: ", confidence_rules) #recursion 
    return confidence_rules
'''
def generate_next_confidence_level(LHS, RHS, freq_k_itemsets, min_confidence):
    for RHS in LHS: #========= for item in LHS
        #RHS = [RHS]
        
        LHS = set(LHS) - set([RHS]) #this in another function
        print("LHS: ", LHS)
        print("RHS: ", RHS) 
        
        LHS  = tuple(LHS)
        #RHS = list(RHS)
        # global candidate_rules
        # nonlocal candidate_rules

        if len(LHS) >= 1:
            confidence = freq_k_itemsets[k][itemset]/ freq_k_itemsets[k-1][LHS] #get support from freq itemsets dictionary
            print("confidence: ", confidence)
            if confidence >= min_confidence:
                candidate_rules.append([LHS, RHS, confidence, freq_k_itemsets[k][itemset]])
    
    
    for rule in candidate_rules:
        print("rule: ", rule)
        if len(rule[0]) > 1:
            generate_next_confidence_level(rule[0], rule[1], freq_k_itemsets, min_confidence)            

    #LHS = tuple(LHS)
    #RHS = tuple(RHS)
    print("candidate rules ",candidate_rules)
    return candidate_rules
    #return candidate_rules
    #generate_next_confidence_level(candidate_rules[0], candidate_rules[1], candidate_rules[2], min_confidence)                    
'''

#----------------------START HERE----------------------
start = timeit.default_timer() #start timer for frequent itemsets

filename = f"{output_file_name}_items_team5.txt" #create itemset file
print("Creating file 1: %s" % filename)
f = open(filename, "w")

F1 = gen_f1(transactions, minimum_support) #generate 1-itemsets
frequent_n_itemset = {} 
frequent_n_itemset[1] = F1 #set 1-itemsets for apriori beginning

k=2 #set k value to generate 2-itemsets

while(len(frequent_n_itemset[k-1]) > 1): #iterate while itemset length is greater than 1
    frequent_n_itemset[k] = {}
    L = candidate_merge(frequent_n_itemset[k-1])  #generate candidate itemsets
    print(f"{k}th level candidate itemsets: {len(L)}")

    L = pruning(L, frequent_n_itemset[k-1], k) #pruning infrequent candidate itemets
    print(f"{k}th level pruned itemsets: {len(L)}")

    frequent_n_itemset[k] = support_count(L, minimum_support, transactions, k) #calculating support count for itemsets
    print(f"{k}th level frequent itemsets: {len(frequent_n_itemset[k])}")
    print(frequent_n_itemset[k])

    for freq_itemset, freq_support in frequent_n_itemset[k].items():  #writing frequent itemsets to file
            clean_string = " ".join(map(str, freq_itemset)) 
            f.write(f"{clean_string} | {freq_support}\n") 
    k = k+1
stop = timeit.default_timer() #stop timer for frequent itemsets

start1 = timeit.default_timer() #start timer for confidence rules

for k in frequent_n_itemset:
    s = candidate_rules(frequent_n_itemset, minimum_confidence) #generate confidence rules

stop1 = timeit.default_timer() #stop timer for confidence rules
print(f"Created {filename}.txt")
f.close()

print("runtime supp=120 conf=0.8: ", (stop-start))         
# ======================================================
# output file 2                                                 
if minimum_confidence != -1:
    filename = f"{output_file_name}_rules_team5.txt" #create file for association rules

    print("Creating file 2: %s" % filename)
    f = open(filename, "w")

    for rule in s:
        clean_string = " ".join(map(str, rule[0]))
        f.write(f"{clean_string} | {rule[1]} | {rule[2]} | {rule[3]}\n") #write rules to file

    f.close()
    print(f"Created {filename}.txt")
else:
    print("confidence = -1, not generating rules file")

    
# ======================================================
# output file 3                                             
filename = f"{output_file_name}_info_team5.txt" #create file for data info

count_item = 0 #counter for items in dataset
for transaction in transactions.values():
    for item in transaction:
        count_item += 1
        
print("Creating file 3: %s" % filename)
f = open(filename, "w")

f.write(
f"""
minsup: {minimum_support} 
minconf: {minimum_confidence}
input file: {input_file_name}
output name: {output_file_name}
Number of items: {count_item}
Number of transactions: {len(transactions)}
Length of largest frequent k-itemset: {k-1}"""
)
for k in frequent_n_itemset: #loop for writing k-level itemsets to file
    if len(frequent_n_itemset[k].keys()) > 0:
        f.write(f"\nNumber of frequent {k}-itemsets: {len(frequent_n_itemset[k].keys())}")

    total_k_itemsets = 0
    for k in frequent_n_itemset:
        total_k_itemsets += len(frequent_n_itemset[k].keys())

highest_confidence = max(s, key=lambda inner_list: inner_list[3]) #finding rule with highest confidence
clean_string = " ".join(map(str, highest_confidence[0]))

f.write(
    f"""
Total number of frequent itemsets: {total_k_itemsets}
Number of high confidence rules: {len(s)}
Rule with highest confidence: {clean_string} | {highest_confidence[1]} | {highest_confidence[2]} | {highest_confidence[3]}
Time to find the frequent itemsets (s): {round((stop-start),2)}
Time to find confident rules (s): {round((stop1-start1),2)}"""
)            

f.close()
#=======================================================================
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

# Initialize dictionaries to store the results
runtimes = {conf: {"itemsets": [], "rules": []} for conf in min_confs}
counts = {conf: {"itemsets": [], "rules": []} for conf in min_confs}

for min_conf in min_confs:
    for min_support_count in min_support_counts:
        min_support = min_support_count / num_transactions

        # Measure runtime for frequent itemset generation
        start_time = time.time()
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
        end_time = time.time()
        runtimes[min_conf]["itemsets"].append(end_time - start_time)

        counts[min_conf]["itemsets"].append(len(frequent_itemsets))

        # Measure runtime for rule creation and count the rules
        rule_count = 0
        start_time = time.time()
        output_file_name = (
            f"output_file_rules_team5_conf_{min_conf}_sup_{min_support_count}.txt"
        )
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
                            rule_count += 1
        end_time = time.time()
        runtimes[min_conf]["rules"].append(end_time - start_time)
        counts[min_conf]["rules"].append(rule_count)

# Now, plot and save the results for both runtimes and counts
for min_conf in min_confs:
    # Plotting runtimes
    plt.figure(figsize=(12, 7))
    bars1 = plt.bar(
        np.array(range(len(min_support_counts))) - 0.2,
        runtimes[min_conf]["itemsets"],
        0.4,
        label="Frequent Itemsets",
        color="blue",
        edgecolor="black",
    )
    bars2 = plt.bar(
        np.array(range(len(min_support_counts))) + 0.2,
        runtimes[min_conf]["rules"],
        0.4,
        label="Rules",
        color="green",
        edgecolor="black",
    )
    for bar in bars1:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.05,
            round(yval, 2),
            ha="center",
            va="bottom",
        )
    for bar in bars2:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.05,
            round(yval, 2),
            ha="center",
            va="bottom",
        )
    plt.xticks(range(len(min_support_counts)), min_support_counts)
    plt.xlabel("Minimum Support Count", fontsize=16)
    plt.ylabel("Time (seconds)", fontsize=16)
    plt.title(f"Runtime for min_conf = {min_conf}", fontsize=18)
    plt.legend(loc="upper left", fontsize=14)
    file_name = f"runtime_plot_min_conf_{min_conf}.png"
    plt.tight_layout()
    plt.savefig(file_name)
    print(f"Runtime plot saved as {file_name}")
    plt.close()

    # Plotting counts
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
    for bar in bars1:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.05,
            round(yval, 2),
            ha="center",
            va="bottom",
        )
    for bar in bars2:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.05,
            round(yval, 2),
            ha="center",
            va="bottom",
        )
    plt.xticks(range(len(min_support_counts)), min_support_counts)
    plt.xlabel("Minimum Support Count", fontsize=16)
    plt.ylabel("Counts", fontsize=16)
    plt.title(f"Counts for min_conf = {min_conf}", fontsize=18)
    plt.legend(loc="upper left", fontsize=14)
    file_name = f"counts_plot_min_conf_{min_conf}.png"
    plt.tight_layout()
    plt.savefig(file_name)
    print(f"Counts plot saved as {file_name}")
    plt.close()

print("Processing completed!")