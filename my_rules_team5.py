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


'''                                                                               #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def generate_k_freq_itemset(transactions, min_support, k, prev_freq_itemsets): #PHASE 2 Responsibilities 7-9 done in this function
    candidate_counts = defaultdict(int)

    # Count the occurrences of each item in the transactions
    for transaction in transactions.values():  # Iterate over the lists of item IDs
        candidate_itemsets = itertools.combinations(transaction, k) # Generate k-itemset candidates 
        for candidate in candidate_itemsets:
            candidate = list(candidate)
            keep_candidate = True
            candidate_subsets = itertools.combinations(candidate, k-1) # Generate every subset in itemset to check support
            for subset in candidate_subsets: 
                subset = str(list(subset))
                keep_candidate = keep_candidate and (subset in prev_freq_itemsets) # If every subset is frequent, keep itemset, else prune

            if keep_candidate: # If survived pruning
                candidate_counts[str(candidate)] += 1 # Preprare for support counting

    
    frequent_k_itemsets = {item: support for item, support in candidate_counts.items() if support >= min_support} # Filter items based on minimum support
    frequent_k_itemsets_sorted = {item: support for item, support in sorted(frequent_k_itemsets.items(), key=lambda x: x[1], reverse=True)} #sort items in descending support count
    return frequent_k_itemsets_sorted  #result is a list of k-itemsets with infrequent itemsets already pruned   
'''


def candidate_merge(prev_freq_itemsets):
    candidate_itemsets = []
    temp_k = len(list(prev_freq_itemsets.keys())[1]) #if k=1

    for i in range(len(prev_freq_itemsets)):#transaction
        for j in range(i + 1, len(prev_freq_itemsets)): #items
            
            itemset1 = list(prev_freq_itemsets.keys())[i]
            itemset2 = list(prev_freq_itemsets.keys())[j]
            if temp_k == 1:
                candidate_itemsets.append(sorted(set(itemset1 + itemset2)))
            
            elif itemset1[:-1] == itemset2[:-1]:
                # Merge (k-1)-itemsets to create a candidate k-itemset
                candidate_itemsets.append(sorted(set(itemset1 + itemset2)))

    return candidate_itemsets 

def pruning(candidate_itemset, k_minus_1_itemset, k):
    pruned_itemsets = []
    

    
        

    return pruned_itemsets

'''
def candidate_rule_generation(): #will be completed in phase 3
    return    
'''
# =============================================================
# output file 1
start = timeit.default_timer()
F1 = gen_f1(transactions, minimum_support)
#F1 = {tuple([1]):10, tuple([2]):20, tuple([3]):30}
filename = f"{output_file_name}_items_team5.txt" #create file 1

print("Creating file 1: %s" % filename)
f = open(filename, "w")
'''
for item, support in F1.items():
    f.write(f"{item} | {support}\n") #write 1-itemsets to file
'''

'''
print(f"Frequent 1-Itemsets (F1) in Descending Order of Support:") #print F1 to console
for item, support in F1.items():
    print(f"{item} | {support}")
'''
    



'''
while len(frequent_n_itemset) > 0: #if itemsets empty, stop
    #frequent_n_plus_1_itemset = generate_k_freq_itemset2(transactions, minimum_support, k, frequent_n_itemset) #apriori algorithm generating k-itemsets
    #frequent_n_plus_1_itemset = generate_k_freq_itemset3(minimum_support, k ,frequent_n_itemset)


    print(f"Frequent {k}-Itemsets (F{k}) in Descending Order of Support:")
    for item, support in frequent_n_plus_1_itemset.items():
        print(f"{item}: {support}") #print Fk to console
        f.write(f"{item} | {support}\n") #write Fk to file.                     FIX ME: only writes up to 2-itemsets
        
    frequent_n_itemset = frequent_n_plus_1_itemset    
    k += 1

f.close()

exit()

'''
#----------------------START HERE----------------------
frequent_n_itemset = {} #set 1-itemsets for apriori beginning
frequent_n_itemset[1] = F1

#frequent_n_itemset = candidate_merge(F1)
k=2 #set k value to generate 2-itemsets


L = candidate_merge(F1)
#print(L)
print(len(frequent_n_itemset[1]))
while(len(frequent_n_itemset[k-1]) > 1):
    frequent_n_itemset[k] = {}
    L = candidate_merge(frequent_n_itemset[k-1]) #pruning input #if subset is infrequent, remove from L
    
    L = pruning(L, frequent_n_itemset[k-1], k)

    print(f"{k}:\n", L)
    


    k += 1
    print("k: ", k)

    
exit()


    #pruning, only append itemsets that survived
    #support count
    #elimination



#rule generation
'''
for k in frequent_n_itemset: 
    for itemset in frequent_n_itemset[k]:
        #generate candidates
'''
#create output file        

stop = timeit.default_timer()

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
#======================================
#generate itemset plot

'''
Things to complete

Phase 1: Complete (update plot code)
Phase 2: Complete
Phase 3: Need to finish

Output file 1:
    -change the format. keeps writing with []
    -only writes up to 2-itemsets

Output file 2:
    -Will be completed in phase 3

Output file 3:
    -length of largest itemset
    -number of k-itemsets
    -everything else

Plot items file:
    -bar plot with the number of frequent k-itemsets for different values of k

Plot rules file:
    -bar plot with the number of high-confidence rules for different values of k


Run code with:    
    -minsup: {50, 100, 150, 200}
    -minconf: {0.8, 0.95} (For each value of minconf, generate two bar charts (or line charts).)
    
'''
