import matplotlib
import matplotlib.pyplot as plt


# Initialize empty dictionary to store the support counts
support_counts = {}

# Read data from the file and populate the support_counts dictionary
with open("test_output_items_team5.txt", "r") as file:
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
