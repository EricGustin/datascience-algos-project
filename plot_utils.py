import matplotlib.pyplot as plt

def scatter_plot(val1, val2, title, x_axis, y_axis):
    """Creates a simple scatter

        Args:
            title: title for graph
            x_axis: x label
            y_axis: y label
            val1: data value 1 for scatter
            val2: data  value 2 for scatter

        Returns:
            graph visualization

        """
    plt.figure()
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.scatter(val1, val2)
    plt.show()


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

def histogram_chart(vals, title, x_axis, y_axis, color):
    """Creates a simple Histogram

        Args:
            title: title for graph
            x_axis: x label
            y_axis: y label
            vals: data values
            color: color for graph visualization

        Returns:
            graph visualization

        """
    plt.figure()
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.hist(vals, bins = 5, color = color)
    plt.show()



