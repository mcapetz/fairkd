import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm

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

auc_diffs = dict()
# -

figs_folder = "figs_final_new"
saved_arrays_folder = "saved_arrays"

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
            teacher_val = np.load(f'{saved_arrays_folder}/auc_ovr_diff_teacher_sbm{dataset}_{seed}_c={class_weights}.npy')
            student_val = np.load(f'{saved_arrays_folder}/auc_ovr_diff_student_sbm{dataset}_{seed}_c={class_weights}.npy')
            teacher_auc = np.load(f'{saved_arrays_folder}/auc_ovr_overall_teacher_sbm{dataset}_{seed}_c={class_weights}.npy')
            student_auc = np.load(f'{saved_arrays_folder}/auc_ovr_overall_student_sbm{dataset}_{seed}_c={class_weights}.npy')
            teacher_acc = np.load(f'{saved_arrays_folder}/acc_teacher_sbm{dataset}_{seed}_c={class_weights}.npy')
            student_acc = np.load(f'{saved_arrays_folder}/acc_student_sbm{dataset}_{seed}_c={class_weights}.npy')
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


def create_bar_plots(datasets, x_label):
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
    plt.savefig(f'{figs_folder}/group_0_1_auc_diff_vs_{x_label[-2:-1]}_all_weights.png')
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
    plt.savefig(f'{figs_folder}/group_0_1_auc_comparison_vs_{x_label[-2:-1]}_all_weights.png')
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
    plt.savefig(f'{figs_folder}/overall_auc_diff_vs_{x_label[-2:-1]}_all_weights.png')
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
    plt.savefig(f'{figs_folder}/overall_acc_diff_vs_{x_label[-2:-1]}_all_weights.png')
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

    cmap_div = cm.get_cmap("RdBu_r")
    
    # Use consistent min/max for colorbar in all heatmaps
    diff_vmin, diff_vmax = global_limits['diff']
    auc_vmin, auc_vmax = global_limits['auc']
    acc_vmin, acc_vmax = global_limits['acc']
    
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
        cbar_kws={'label': 'AUC Difference (Teacher-Student)'},
        vmin=diff_vmin,  # Set consistent limits
        vmax=diff_vmax
    )
    plt.title('Group 0-1 AUC Difference by Class Weight and '+ x_label, fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('Class Weights', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{figs_folder}/group_0_1_auc_diff_{x_label[-2:-1]}_heatmap.png')
    plt.close()
    
    # 2. Heatmap for Overall AUC Difference
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        overall_auc_matrix,
        annot=True,
        fmt=".3f",
        cmap=cmap_div,
        center=0,
        linewidths=.5,
        xticklabels=datasets,
        yticklabels=class_weights_list,
        cbar_kws={'label': 'Overall AUC Difference'},
        vmin=auc_vmin,  # Set consistent limits
        vmax=auc_vmax
    )
    plt.title('Overall AUC Difference by Class Weight and '+ x_label, fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('Class Weights', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{figs_folder}/overall_auc_diff_{x_label[-2:-1]}_heatmap.png')
    plt.close()
    
    # 3. Heatmap for Overall Accuracy Difference
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        overall_acc_matrix,
        annot=True,
        fmt=".3f",
        cmap=cmap_div,
        center=0,
        linewidths=.5,
        xticklabels=datasets,
        yticklabels=class_weights_list,
        cbar_kws={'label': 'Overall Accuracy Difference'},
        vmin=acc_vmin,  # Set consistent limits
        vmax=acc_vmax
    )
    plt.title('Overall Accuracy Difference by Class Weight and '+ x_label, fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('Class Weights', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{figs_folder}/overall_acc_diff_{x_label[-2:-1]}_heatmap.png')
    plt.close()


def create_line_plots(datasets, x_label):
    auc_ts_vmin, auc_ts_vmax = global_limits['auc_ts']
    
    # Define colors for different class weights
    colors = plt.cm.viridis(np.linspace(0, 1, len(class_weights_list)))

    # Plot Group 0-1 AUC Comparison for Teacher and Student as line plots
    plt.figure(figsize=(15, 10))
    
    # Set up x positions
    x_positions = np.arange(len(datasets))
    
    for i, class_weights in enumerate(class_weights_list):
        avg_teacher = [auc_diffs[dataset][class_weights]['avg_teacher'] for dataset in datasets]
        std_teacher = [auc_diffs[dataset][class_weights]['std_teacher'] for dataset in datasets]

        avg_student = [auc_diffs[dataset][class_weights]['avg_student'] for dataset in datasets]
        std_student = [auc_diffs[dataset][class_weights]['std_student'] for dataset in datasets]

        # Plot teacher line with error bands
        plt.plot(x_positions, avg_teacher, '-o', 
                 label=f'Teacher {class_weights}',
                 color=colors[i],
                 linewidth=2,
                 markersize=8)
        
        plt.fill_between(x_positions, 
                         np.array(avg_teacher) - np.array(std_teacher)/np.sqrt(len(seeds)),
                         np.array(avg_teacher) + np.array(std_teacher)/np.sqrt(len(seeds)),
                         color=colors[i], alpha=0.2)

        # Plot student line with error bands
        plt.plot(x_positions, avg_student, '--s', 
                 label=f'Student {class_weights}',
                 color=colors[i], 
                 linewidth=2,
                 markersize=8)
        
        plt.fill_between(x_positions, 
                         np.array(avg_student) - np.array(std_student)/np.sqrt(len(seeds)),
                         np.array(avg_student) + np.array(std_student)/np.sqrt(len(seeds)),
                         color=colors[i], alpha=0.1)

    plt.ylim(auc_ts_vmin, auc_ts_vmax)
    plt.title('Group 0-1 AUC Comparison vs '+ x_label)
    plt.xlabel(x_label)
    plt.ylabel('Group 0-1 AUC')
    plt.xticks(x_positions, datasets)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f'{figs_folder}/group_0_1_auc_comparison_vs_{x_label[-2:-1]}_line.png')
    plt.close()


# +
def create_teacher_plot(datasets, x_label):
    auc_ts_vmin, auc_ts_vmax = global_limits['auc_ts']
    # Define colors for different class weights
    colors = plt.cm.viridis(np.linspace(0, 1, len(class_weights_list)))
    # Plot Group 0-1 AUC Comparison for Teacher only as line plots
    plt.figure(figsize=(15, 10))
    
    # Set up x positions
    x_positions = np.arange(len(datasets))
    
    for i, class_weights in enumerate(class_weights_list):
        avg_teacher = [auc_diffs[dataset][class_weights]['avg_teacher'] for dataset in datasets]
        std_teacher = [auc_diffs[dataset][class_weights]['std_teacher'] for dataset in datasets]
        
        # Plot teacher line with error bands
        plt.plot(x_positions, avg_teacher, '-o', 
                 label=f'Teacher {class_weights}',
                 color=colors[i],
                 linewidth=2,
                 markersize=8)
        
        plt.fill_between(x_positions, 
                         np.array(avg_teacher) - np.array(std_teacher)/np.sqrt(len(seeds)),
                         np.array(avg_teacher) + np.array(std_teacher)/np.sqrt(len(seeds)),
                         color=colors[i], alpha=0.2)
    
    plt.ylim(auc_ts_vmin, auc_ts_vmax)
    plt.title('Teacher Model: Group 0-1 AUC Comparison vs ' + x_label)
    plt.xlabel(x_label)
    plt.ylabel('Group 0-1 AUC')
    plt.xticks(x_positions, datasets)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f'{figs_folder}/teacher_group_0_1_auc_comparison_vs_{x_label[-2:-1]}_line.png')
    plt.close()

def create_student_plot(datasets, x_label):
    auc_ts_vmin, auc_ts_vmax = global_limits['auc_ts']
    # Define colors for different class weights
    colors = plt.cm.viridis(np.linspace(0, 1, len(class_weights_list)))
    # Plot Group 0-1 AUC Comparison for Student only as line plots
    plt.figure(figsize=(15, 10))
    
    # Set up x positions
    x_positions = np.arange(len(datasets))
    
    for i, class_weights in enumerate(class_weights_list):
        avg_student = [auc_diffs[dataset][class_weights]['avg_student'] for dataset in datasets]
        std_student = [auc_diffs[dataset][class_weights]['std_student'] for dataset in datasets]
        
        # Plot student line with different style
        plt.plot(x_positions, avg_student, '--s', 
                 label=f'Student {class_weights}',
                 color=colors[i], 
                 linewidth=2,
                 markersize=8)
        
        plt.fill_between(x_positions, 
                         np.array(avg_student) - np.array(std_student)/np.sqrt(len(seeds)),
                         np.array(avg_student) + np.array(std_student)/np.sqrt(len(seeds)),
                         color=colors[i], alpha=0.2)
    
    plt.ylim(auc_ts_vmin, auc_ts_vmax)
    plt.title('Student Model: Group 0-1 AUC Comparison vs ' + x_label)
    plt.xlabel(x_label)
    plt.ylabel('Group 0-1 AUC')
    plt.xticks(x_positions, datasets)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f'{figs_folder}/student_group_0_1_auc_comparison_vs_{x_label[-2:-1]}_line.png')
    plt.close()


# +
# First, determine global min and max values across all datasets
def find_global_min_max(datasets_list):
    global_min_diff = float('inf')
    global_max_diff = float('-inf')
    global_min_auc = float('inf')
    global_max_auc = float('-inf')
    global_min_acc = float('inf')
    global_max_acc = float('-inf')
    global_min_auc_teacher_student = float('inf')
    global_max_auc_teacher_student = float('-inf')
    
    for datasets_type in datasets_list:
        for dataset in datasets_type:
            for class_weight in class_weights_list:
                # For group differences
                diff_val = auc_diffs[dataset][class_weight]['avg_diff']
                global_min_diff = min(global_min_diff, diff_val)
                global_max_diff = max(global_max_diff, diff_val)
                
                # For overall AUC values
                auc_val = auc_diffs[dataset][class_weight]['avg_aucs']
                global_min_auc = min(global_min_auc, auc_val)
                global_max_auc = max(global_max_auc, auc_val)
                
                # For teacher/student AUC values
                avg_teacher = auc_diffs[dataset][class_weight]['avg_teacher']
                avg_student = auc_diffs[dataset][class_weight]['avg_student']

                # Update global min and max
                global_min_auc_teacher_student = min(global_min_auc_teacher_student, avg_teacher, avg_student)
                global_max_auc_teacher_student = max(global_max_auc_teacher_student, avg_teacher, avg_student)

                # For Accuracy values
                acc_val = auc_diffs[dataset][class_weight]['avg_accs']
                global_min_acc = min(global_min_acc, acc_val)
                global_max_acc = max(global_max_acc, acc_val)
    
    # Add some padding to the limits (e.g., 5%)
    padding_diff = (global_max_diff - global_min_diff) * 0.05
    padding_auc = (global_max_auc - global_min_auc) * 0.05
    padding_acc = (global_max_acc - global_min_acc) * 0.05
    padding_auc_ts = (global_max_auc_teacher_student - global_min_auc_teacher_student) * 0.05
    
    return {
        'diff': (global_min_diff - padding_diff, global_max_diff + padding_diff),
        'auc': (global_min_auc - padding_auc, global_max_auc + padding_auc),
        'acc': (global_min_acc - padding_acc, global_max_acc + padding_acc),
        'auc_ts': (global_min_auc_teacher_student - padding_auc_ts, global_max_auc_teacher_student + padding_auc_ts)
        
    }

# Calculate global limits before creating any plots
global_limits = find_global_min_max([datasets_p, datasets_q])
# -

for (dataset, x_label) in [(datasets_p, 'Group Balance (p)'),(datasets_q, 'Edge probability Ratio (q)')]:
    create_bar_plots(dataset, x_label)
    create_heatmaps(dataset, x_label)
    create_line_plots(dataset, x_label)
    create_teacher_plot(dataset, x_label)
    create_student_plot(dataset, x_label)
