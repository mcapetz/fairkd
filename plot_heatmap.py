import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

datasets_p = ["0.1", "0.2", "0.3", "0.4", "0.5"] # these are the p values
datasets_q = ["0.05", "0.25", "0.50", "0.75", "1.00"] # these are the p values
datasets = datasets_p + datasets_q
seeds = ["0", "1", "2", "3", "4"]
class_weights_list = ["0.1,0.9", "0.2,0.8", "0.3,0.7", "0.4,0.6", "0.5,0.5"]

# +
max_diff = float('-inf')
min_diff = float('inf')

max_auc = float('-inf')
min_auc = float('inf')
# -

auc_diffs = dict()

# load in the auc values
for dataset in datasets:  
    
    auc_diffs[dataset] = dict()
    
    teachers = []
    students = []
    diffs = []
    aucs = []
    accs = []
    
    for class_weights in class_weights_list:

        # avg and std for seeds
        for seed in seeds: 
            teacher_val = np.load(f'saved_arrays/auc_ovr_diff_teacher_sbm{dataset}_{seed}_c={class_weights}.npy')
            student_val = np.load(f'saved_arrays/auc_ovr_diff_student_sbm{dataset}_{seed}_c={class_weights}.npy')
            teacher_auc = np.load(f'saved_arrays/auc_ovr_overall_teacher_sbm{dataset}_{seed}_c={class_weights}.npy')
            student_auc = np.load(f'saved_arrays/auc_ovr_overall_student_sbm{dataset}_{seed}_c={class_weights}.npy')
            teacher_acc = np.load(f'saved_arrays/acc_teacher_sbm{dataset}_{seed}_c={class_weights}.npy')
            student_acc = np.load(f'saved_arrays/acc_student_sbm{dataset}_{seed}_c={class_weights}.npy')
            diff = teacher_val - student_val
            auc = teacher_auc-student_auc
            acc = teacher_acc-student_acc

            max_diff = max(max_diff, diff)
            min_diff = min(min_diff, diff)
            max_auc = max(max_auc, teacher_val, student_val)
            min_auc = min(min_auc, teacher_val, student_val)

            teachers.append(teacher_val)
            students.append(student_val)
            diffs.append(diff)
            aucs.append(auc)
            accs.append(acc)

        avg_teacher = np.mean(teachers, axis=0)
        std_teacher = np.std(teachers, axis=0)

        avg_student = np.mean(students, axis=0)
        std_student = np.std(students, axis=0)

        avg_diff = np.mean(diffs, axis=0)
        std_diff = np.std(diffs, axis=0)

        avg_auc = np.mean(aucs, axis=0)
        std_auc = np.std(aucs, axis=0)

        avg_acc = np.mean(accs, axis=0)
        std_acc = np.std(accs, axis=0)
        
        auc_diffs[dataset][class_weights] = {
            "avg_teacher": np.mean(teachers, axis=0),
            "std_teacher": np.std(teachers, axis=0),
            "avg_student": np.mean(students, axis=0),
            "std_student": np.std(students, axis=0),
            "avg_diff": np.mean(diffs, axis=0),
            "std_diff": np.std(diffs, axis=0),
            "avg_aucs": np.mean(aucs, axis=0),
            "std_aucs": np.std(aucs, axis=0),
            "avg_accs": np.mean(accs, axis=0),
            "std_accs": np.std(accs, axis=0),
        }


def plot_figs(datasets, x_label):
    # Define colors for different class weights
    colors = plt.cm.viridis(np.linspace(0, 1, len(class_weights_list)))

    # 1. Plot Group 0-1 AUC Difference between Teacher and Student
    plt.figure(figsize=(12, 7))
    bar_width = 0.15
    index = np.arange(len(datasets))

    for i, class_weights in enumerate(class_weights_list):
        avg_diffs = [auc_diffs[dataset][class_weights]['avg_diff'] for dataset in datasets]
        std_diffs = [auc_diffs[dataset][class_weights]['std_diff'] for dataset in datasets]

        plt.bar(index + i*bar_width - bar_width*2, avg_diffs, bar_width, 
                yerr=np.array(std_diffs)/np.sqrt(len(seeds)), 
                capsize=3, 
                label=f'Class Weights: {class_weights}',
                color=colors[i])

    plt.title('Group 0-1 AUC Difference (Teacher-Student) vs ' + x_label)
    plt.xlabel(x_label)
    plt.ylabel('AUC Difference (Teacher-Student)')
    plt.xticks(index, datasets)
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f'figs_final/group_0_1_auc_diff_vs_{x_label[-2:-1]}_all_weights.png')
    plt.close()

    # 2. Plot Group 0-1 AUC Comparison for Teacher and Student
    plt.figure(figsize=(15, 10))
    bar_width = 0.08
    index = np.arange(len(datasets))

    for i, class_weights in enumerate(class_weights_list):
        avg_teacher = [auc_diffs[dataset][class_weights]['avg_teacher'] for dataset in datasets]
        std_teacher = [auc_diffs[dataset][class_weights]['std_teacher'] for dataset in datasets]

        avg_student = [auc_diffs[dataset][class_weights]['avg_student'] for dataset in datasets]
        std_student = [auc_diffs[dataset][class_weights]['std_student'] for dataset in datasets]

        plt.bar(index + i*bar_width*2 - bar_width*4, avg_teacher, bar_width, 
                yerr=np.array(std_teacher)/np.sqrt(len(seeds)), 
                capsize=3, 
                label=f'Teacher {class_weights}',
                color=colors[i])

        plt.bar(index + i*bar_width*2 - bar_width*4 + bar_width, avg_student, bar_width, 
                yerr=np.array(std_student)/np.sqrt(len(seeds)), 
                capsize=3, 
                label=f'Student {class_weights}',
                color=colors[i], alpha=0.5)

    plt.title('Group 0-1 AUC Comparison vs '+ x_label)
    plt.xlabel(x_label)
    plt.ylabel('Group 0-1 AUC')
    plt.xticks(index, datasets)
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f'figs_final/group_0_1_auc_comparison_vs_{x_label[-2:-1]}_all_weights.png')
    plt.close()

    # 3. Plot Overall AUC Difference
    plt.figure(figsize=(12, 7))
    bar_width = 0.15
    index = np.arange(len(datasets))

    for i, class_weights in enumerate(class_weights_list):
        avg_diffs = [auc_diffs[dataset][class_weights]['avg_aucs'] for dataset in datasets]
        std_diffs = [auc_diffs[dataset][class_weights]['std_aucs'] for dataset in datasets]

        plt.bar(index + i*bar_width - bar_width*2, avg_diffs, bar_width, 
                yerr=np.array(std_diffs)/np.sqrt(len(seeds)), 
                capsize=3, 
                label=f'Class Weights: {class_weights}',
                color=colors[i])

    plt.title('Overall AUC Difference (Teacher-Student) vs '+ x_label)
    plt.xlabel(x_label)
    plt.ylabel('Overall AUC Difference (Teacher-Student)')
    plt.xticks(index, datasets)
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f'figs_final/overall_auc_diff_vs_{x_label[-2:-1]}_all_weights.png')
    plt.close()

    # 4. Plot Overall Accuracy Difference
    plt.figure(figsize=(12, 7))
    bar_width = 0.15
    index = np.arange(len(datasets))

    for i, class_weights in enumerate(class_weights_list):
        avg_diffs = [auc_diffs[dataset][class_weights]['avg_accs'] for dataset in datasets]
        std_diffs = [auc_diffs[dataset][class_weights]['std_accs'] for dataset in datasets]

        plt.bar(index + i*bar_width - bar_width*2, avg_diffs, bar_width, 
                yerr=np.array(std_diffs)/np.sqrt(len(seeds)), 
                capsize=3, 
                label=f'Class Weights: {class_weights}',
                color=colors[i])

    plt.title('Overall Accuracy Difference (Teacher-Student) vs '+ x_label)
    plt.xlabel(x_label)
    plt.ylabel('Overall Accuracy Difference (Teacher-Student)')
    plt.xticks(index, datasets)
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f'figs_final/overall_acc_diff_vs_{x_label[-2:-1]}_all_weights.png')
    plt.close()


def create_heatmaps(datasets, x_label):
    # Prepare data structures for heatmaps
    group_diff_matrix = np.zeros((len(class_weights_list), len(datasets)))
    overall_auc_matrix = np.zeros((len(class_weights_list), len(datasets)))
    overall_acc_matrix = np.zeros((len(class_weights_list), len(datasets)))
    
    # Populate matrices
    for i, class_weight in enumerate(class_weights_list):
        for j, p_val in enumerate(datasets):
            group_diff_matrix[i, j] = auc_diffs[p_val][class_weight]['avg_diff']
            overall_auc_matrix[i, j] = auc_diffs[p_val][class_weight]['avg_aucs']
            overall_acc_matrix[i, j] = auc_diffs[p_val][class_weight]['avg_accs']
    
    # Custom diverging colormap centered at zero
    colors = ["darkblue", "royalblue", "white", "lightcoral", "darkred"]
    cmap_div = LinearSegmentedColormap.from_list("custom_div", colors, N=256)
    
    # 1. Heatmap for Group 0-1 AUC Difference
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        group_diff_matrix,
        annot=True,
        fmt=".3f",
        cmap=cmap_div,
        center=0,
        linewidths=.5,
        xticklabels=datasets,
        yticklabels=class_weights_list,
        cbar_kws={'label': 'AUC Difference (Teacher-Student)'}
    )
    plt.title('Group 0-1 AUC Difference by Class Weight and '+ x_label, fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('Class Weights', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'figs_final/group_0_1_auc_diff_{x_label[-2:-1]}_heatmap.png')
    plt.close()
    
    # 2. Heatmap for Overall AUC Difference
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        overall_auc_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        linewidths=.5,
        xticklabels=datasets,
        yticklabels=class_weights_list,
        cbar_kws={'label': 'Overall AUC Difference'}
    )
    plt.title('Overall AUC Difference by Class Weight and '+ x_label, fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('Class Weights', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'figs_final/overall_auc_diff_{x_label[-2:-1]}_heatmap.png')
    plt.close()
    
    # 3. Heatmap for Overall Accuracy Difference
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        overall_acc_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        linewidths=.5,
        xticklabels=datasets,
        yticklabels=class_weights_list,
        cbar_kws={'label': 'Overall Accuracy Difference'}
    )
    plt.title('Overall Accuracy Difference by Class Weight and '+ x_label, fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('Class Weights', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'figs_final/overall_acc_diff_{x_label[-2:-1]}_heatmap.png')
    plt.close()


# plot_figs(datasets_p, 'Group Balance (p)')
# plot_figs(datasets_q, 'Ratio between intra/inter edge probability (q)')
create_heatmaps(datasets_p, 'Group Balance (p)')
create_heatmaps(datasets_q, 'Ratio between intra/inter edge probability (q)')
