import matplotlib.pyplot as plt



def plot_loss_and_accuracy(losses, accuracies):
    """
    Plots the loss and accuracy over epochs on the same graph with separate y-axes,
    using a dark background and pastel colors.

    Parameters:
    losses (list or array-like): List or array of loss values over time.
    accuracies (list or array-like): List or array of accuracy values over time.
    epochs (int): Number of epochs (should match the length of losses and accuracies).
    """
    epochs = len(losses) 
    # Set dark background style
    plt.style.use('dark_background')

    # Create a figure and an axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot loss with pastel red
    color = '#AF3F31'  # Pastel red
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(range(epochs), losses, color=color, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.7)  # Light grid lines

    # Create a second y-axis to plot accuracy
    ax2 = ax1.twinx()
    color = '#4B5B95'  # Pastel purple
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(range(epochs), accuracies, color=color, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    # Title and legends
    plt.title('Loss and Accuracy over Epochs', color='white')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Show plot
    plt.tight_layout()
    
    return plt


def format_time(seconds):
    """
    Converts a time in seconds into a readable string with the largest appropriate unit.

    Parameters:
    seconds (float or int): Time duration in seconds.

    Returns:
    str: Readable string representing the time.
    """
    if seconds < 0:
        raise ValueError("Time cannot be negative")
    
    # Define time units
    units = [
        ('day', 86400),   # 24 * 60 * 60
        ('hour', 3600),   # 60 * 60
        ('minute', 60),
        ('second', 1)
    ]

    # Convert time into the largest unit
    for unit_name, unit_seconds in units:
        if seconds >= unit_seconds:
            value = seconds / unit_seconds
            if unit_name == 'second':
                # Rounding to the nearest integer for seconds
                return f"{round(value)} {unit_name}{'s' if round(value) != 1 else ''}"
            else:
                # Rounding to two decimal places for other units
                return f"{value:.2f} {unit_name}{'s' if value != 1 else ''}"


