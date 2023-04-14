import matplotlib.pyplot as plt


def plot_log_times(df):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df[['start_timestamp', 'complete_timestamp']], df['case:concept:name'], alpha=0.5,
            label=['Start_date', 'End_date'])

    plt.legend()
    plt.show()


def plot_only_lines(df):
    fig, ax = plt.subplots(figsize=[14, 6])
    for i in range(df.shape[0] - 1):
        ax.plot([df['start_timestamp'][i], df['complete_timestamp'][i]],
                [df['case:concept:name'][i], df['case:concept:name'][i]], 'r-')
    ax.get_yaxis().set_visible(False)
    plt.show()
