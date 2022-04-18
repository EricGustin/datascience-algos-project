import matplotlib.pyplot as plt


def create_frequency_diagram(x_values, y_values, title, x_label, y_label):
    """
    This function creates a frequency diagram with matplotlib

    Parameters:
    -----------
    x (list)
    y (list)
    title (str)
    x_label (str)
    y_label (str)
    """
    fig = plt.figure(figsize=(20, 7))
    axes = fig.add_axes([0, 0, 1, 1])
    axes.bar([str(val) for val in x_values], y_values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()