import matplotlib.pyplot as plt

# Data setup
k_values, counts = (1, 2, 3), (5, 1, 2)
k_values_2, counts_2 = k_values, counts
bar_width = 0.35
r1 = range(len(counts))
r2 = [x + bar_width for x in r1]

# Colors and legend details
colors_data1 = "red"
colors_data2 = "lightblue"
legend_labels = [
    "Data number 1 (Support: 50, Confidence: 0.8)",
    "Data number 2 (Support: 50, Confidence: 0.8)",
]

# Creating the side-by-side bar plot
plt.figure(figsize=(12, 9))
bars1 = plt.bar(
    r1,
    counts,
    width=bar_width,
    color=colors_data1,
    edgecolor="black",
    label=legend_labels[0],
)
bars2 = plt.bar(
    r2,
    counts_2,
    width=bar_width,
    color=colors_data2,
    edgecolor="black",
    label=legend_labels[1],
)

# Adding annotations above each bar
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2 - 0.15,
            height + 0.2,
            str(int(height)),
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=14,
        )

# Setting labels, title, and other parameters
plt.xlabel("Number of Antecedent Items (k)", fontsize=15)
plt.ylabel("Number of High-Confidence Rules", fontsize=15)
plt.title(
    "Comparative Number of High-Confidence Rules for Different Values of k", fontsize=16
)
plt.xticks([r + bar_width for r in range(len(counts))], k_values, fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(0, max(max(counts), max(counts_2)) + 2)
plt.legend(loc="upper right", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Saving the plot
output_file_name = "<output_file_name>_plot_rules_team5.png"
plt.tight_layout()
plt.savefig(output_file_name)
plt.show()
