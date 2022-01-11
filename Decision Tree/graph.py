import matplotlib.pyplot as plt

def graphing(withPruning, withoutPruning):
    """For each 10 increamented data entry, graph its accuracy."""
    plt.plot(withPruning, label="With pruning")
    plt.plot(withoutPruning, label="Without pruning")
    # Graph settings
    plt.title("Training accuracy")
    plt.xlabel("Size")
    plt.ylabel("Accuracy")
    plt.legend(["With pruning", "Without pruning"], loc="lower right")
    plt.show()


# graphing(
#     [10, 20, 30, 40, 50],
#     [86, 87, 88, 89, 95],
#     [10, 20, 30, 40, 50],
#     [95, 96, 97, 98, 99],
# )
