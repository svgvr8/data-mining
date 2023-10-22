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
    confidence_rules = [] #make dict, keys=len(RHS), values=list of RHS,LHS,support, confidence, 
    #might need to define in main program
    #X -> Y - X
    #[ABCD]
    #ABC => D
    #ABD => C

    for k in freq_k_itemsets:
        if k >= 2: #move loops to main
            for itemset in freq_k_itemsets[k]:
                
                            #call next level function
                for RHS in itemset: #========= for item in LHS
                    #itemset = list(itemset)
                    LHS = set(itemset) - set([RHS]) #this in another function
                    print("LHS: ", LHS)
                    print("RHS: ", RHS) #<==================
                                                            #generate next level(LHS, RHS(empty), freq k itemsets) (first time is itemsets, )
                                                            #check if they have enough confidence
                                                            #if true, save to confidence_rules
                                                            #call same function() using an iterable to generate next level
                    LHS = tuple(LHS)
                    if len(LHS) >= 1:
                        confidence = freq_k_itemsets[k][itemset]/ freq_k_itemsets[k-1][LHS] #get support from freq itemsets dictionary
                        support = freq_k_itemsets[k][itemset]/len(transactions)
                        print("confidence: ", confidence)
                        if confidence >= min_confidence:
                            confidence_rules.append([LHS, RHS,round(support,2), round(confidence,2)]) #, freq_k_itemsets[k][itemset]

                    #first function should move 1 item to other side and compute confidence
                    #print(itemset)
                    #itemset = list(itemset)
                    '''
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
                    '''
    
    print("confidence rules: ", confidence_rules) #recursion   
    #maybe make secondary function 
    return confidence_rules

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

def extract_k_counts(filename):
    k_counts = defaultdict(int)
    with open(filename, "r") as file:
        for line in file:
            items = line.split("|")[0].strip()[1:-1].split(", ")
            k = len(items)
            k_counts[k] += 1
    return k_counts

#----------------------START HERE----------------------
start = timeit.default_timer()
#create files
filename = f"{output_file_name}_items_team5.txt" #create file 1
print("Creating file 1: %s" % filename)
f = open(filename, "w")

F1 = gen_f1(transactions, minimum_support)
frequent_n_itemset = {} #set 1-itemsets for apriori beginning
frequent_n_itemset[1] = F1

k=2 #set k value to generate 2-itemsets

#candidate_rules = []

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
            clean_string = " ".join(map(str, freq_itemset))
            f.write(f"{clean_string} | {freq_support}\n")
    
    # s = candidate_rules(frequent_n_itemset, minimum_confidence)
    # print(s)
    k = k+1

start1 = timeit.default_timer()    
for k in frequent_n_itemset:
    s = candidate_rules(frequent_n_itemset, minimum_confidence)
stop1 = timeit.default_timer()
# for k in frequent_n_itemset:
#         if k >= 2: #move loops to main
#             for itemset in frequent_n_itemset[k]:
#                 generate_next_confidence_level(itemset, set(), frequent_n_itemset, minimum_confidence)
print(f"Created {filename}.txt")
f.close()
stop = timeit.default_timer()

print("runtime supp=50 conf=0.8: ", (stop-start))

# for k in frequent_n_itemset:
#     if k >= 2:
#         for itemset in frequent_n_itemset[k]:
#             print(itemset)
#             s = candidate_rules(itemset, minimum_confidence)
      
# print("s",s)            
# ======================================================
# output file 2                                                 Will be completed in Phase 3
if minimum_confidence != -1:
    filename = f"{output_file_name}_rules_team5.txt"

    print("Creating file 2: %s" % filename)
    f = open(filename, "w")

    for rule in s:
        clean_string = " ".join(map(str, rule[0]))
        f.write(f"{clean_string} | {rule[1]} | {rule[2]} | {rule[3]}\n")

    f.close()
    print(f"Created {filename}.txt")
else:
    print("confidence = -1, not generating rules file")

    
# ======================================================
# output file 3                                             Will be completed in Phase 3
filename = f"{output_file_name}_info_team5.txt"

count_item = 0
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
for k in frequent_n_itemset:
    if len(frequent_n_itemset[k].keys()) > 0:
        f.write(f"\nNumber of frequent {k}-itemsets: {len(frequent_n_itemset[k].keys())}")

    total_k_itemsets = 0
    for k in frequent_n_itemset:
        total_k_itemsets += len(frequent_n_itemset[k].keys())

highest_confidence = max(s, key=lambda inner_list: inner_list[3])
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
exit()
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