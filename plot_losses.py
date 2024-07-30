import csv

import matplotlib.pyplot as plt

colors = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
    '#17becf'   # Cyan
]


def read_csv_to_dict(file_path):
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        data = {}
        for row in reader:
            for column, value in row.items():
                if column not in data:
                    data[column] = []
                value = float(value) if value != '' else None
                data[column].append(value)
    return data


if __name__ == "__main__":

    desc = "TRADES"

    orig_data_dict = read_csv_to_dict(f'./trained_models/WRN28-10Swish_cifar10_lr0p2_{desc}_epoch100_bs512/stats_adv.csv')
    syn_data_dict = read_csv_to_dict(f'./trained_models/WRN28-10Swish_cifar10s_Syn30k_lr0p2_{desc}_epoch100_bs512_fraction0p2_ls0p1/stats_adv.csv')
    ce_data_dict = read_csv_to_dict(f'./trained_models/WRN28-10Swish_cifar10s_CE30k_lr0p2_{desc}_epoch100_bs512_fraction0p2_ls0p1/stats_adv.csv')


    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the train accuracy, robust train accuracy, val robust accuracy, standard test accuracy
    for i, key in enumerate(["train_clean_acc", "test_clean_acc"]):
        ls = '-' if key == "train_clean_acc" else '--'
        if key in orig_data_dict.keys():
            ax.plot(orig_data_dict['epoch'], orig_data_dict[key], label=f'{desc}: ' + key, color=colors[0], linestyle=ls, linewidth=2.5)
        # if key in syn_data_dict.keys():
        #     ax.plot(syn_data_dict['epoch'], syn_data_dict[key], label=f'{desc} + EDM: ' + key, color=colors[2], linestyle=ls, linewidth=2.5)
        if key in ce_data_dict.keys():
            ax.plot(ce_data_dict['epoch'], ce_data_dict[key], label=f'{desc} + CE: ' + key, color=colors[1], linestyle=ls, linewidth=2.5)
        
    # Add titles and labels
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    # Customize the grid
    ax.grid(True, which='both')

    # Add a legend
    ax.legend()

    fig.savefig(f'./figs/accuracy_clean_{desc}.png')


    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the train accuracy, robust train accuracy, val robust accuracy, standard test accuracy
    for i, key in enumerate(["train_adversarial_acc", "eval_adversarial_acc"]):
        ls = '-' if key == "train_adversarial_acc" else '--'
        if key in orig_data_dict.keys():
            ax.plot(orig_data_dict['epoch'], orig_data_dict[key], label=f'{desc}: ' + key, color=colors[0], linestyle=ls, linewidth=2.5)
        # if key in syn_data_dict.keys():
        #     ax.plot(syn_data_dict['epoch'], syn_data_dict[key], label=f'{desc} + EDM: ' + key, color=colors[2], linestyle=ls, linewidth=2.5)
        if key in ce_data_dict.keys():
            ax.plot(ce_data_dict['epoch'], ce_data_dict[key], label=f'{desc} + CE: ' + key, color=colors[1], linestyle=ls, linewidth=2.5)
        
    # Add titles and labels
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    # Customize the grid
    ax.grid(True, which='both')

    # Add a legend
    ax.legend()

    fig.savefig(f'./figs/accuracy_adv_{desc}.png')