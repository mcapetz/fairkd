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

# +
figs_folder = "figs_final_trad"
saved_arrays_folder = "saved_arrays"

import os

os.makedirs(figs_folder, exist_ok=True)
# -

# load in the auc values
for dataset in datasets:  
    
    auc_diffs[dataset] = dict()
    
    teachers_dp = []
    students_dp = []
    diffs_dp = []
    
    teachers_dp_group = []
    students_dp_group = []
    diffs_dp_group = []
    
    teachers_eo = []
    students_eo = []
    diffs_eo = []
    
    teachers_eo_group = []
    students_eo_group = []
    diffs_eo_group = []
    
    
    for class_weights in class_weights_list:

        # avg and std for seeds
        for seed in seeds: 
            dp_0, dp_1, eo_0, eo_1, dp, eo, acc_diff = np.load(f'{saved_arrays_folder}/trad_metrics_teacher_sbm{dataset}_{seed}_c={class_weights}.npy')
            dp_0_, dp_1_, eo_0_, eo_1_, dp_, eo_, acc_diff = np.load(f'{saved_arrays_folder}/trad_metrics_student_sbm{dataset}_{seed}_c={class_weights}.npy')

            teachers_dp.append(dp)
            students_dp.append(dp_)
            diffs_dp.append(dp-dp_)
            
            teachers_dp_group.append(dp_0 - dp_1)
            students_dp_group.append(dp_0_ - dp_1_)
            diffs_dp_group.append((dp_0 - dp_1)-(dp_0_ - dp_1_))
            
            teachers_eo.append(eo)
            students_eo.append(eo_)
            diffs_eo.append(eo-eo_)
            
            teachers_eo_group.append(eo_0 - eo_1)
            students_eo_group.append(eo_0_ - eo_1_)
            diffs_eo_group.append((eo_0 - eo_1)-(eo_0_ - eo_1_))
        
        
        auc_diffs[dataset][class_weights] = {
            "avg_teachers_dp" : np.mean(teachers_dp, axis=0),
            "std_teachers_dp" : np.std(teachers_dp, axis=0),
            "avg_students_dp" : np.mean(students_dp, axis=0),
            "std_students_dp" : np.std(students_dp, axis=0),
            "avg_diffs_dp" : np.mean(diffs_dp, axis=0),
            "std_diffs_dp" : np.std(diffs_dp, axis=0),

            "avg_teachers_dp_group" : np.mean(teachers_dp_group, axis=0),
            "std_teachers_dp_group" : np.std(teachers_dp_group, axis=0),
            "avg_students_dp_group" : np.mean(students_dp_group, axis=0),
            "std_students_dp_group" : np.std(students_dp_group, axis=0),
            "avg_diffs_dp_group" : np.mean(diffs_dp_group, axis=0),
            "std_diffs_dp_group" : np.std(diffs_dp_group, axis=0),

            "avg_teachers_eo" : np.mean(teachers_eo, axis=0),
            "std_teachers_eo" : np.std(teachers_eo, axis=0),
            "avg_students_eo" : np.mean(students_eo, axis=0),
            "std_students_eo" : np.std(students_eo, axis=0),
            "avg_diffs_eo" : np.mean(diffs_eo, axis=0),
            "std_diffs_eo" : np.std(diffs_eo, axis=0),

            "avg_teachers_eo_group" : np.mean(teachers_eo_group, axis=0),
            "std_teachers_eo_group" : np.std(teachers_eo_group, axis=0),
            "avg_students_eo_group" : np.mean(students_eo_group, axis=0),
            "std_students_eo_group" : np.std(students_eo_group, axis=0),
            "avg_diffs_eo_group" : np.mean(diffs_eo_group, axis=0),
            "std_diffs_eo_group" : np.std(diffs_eo_group, axis=0)
        }


def create_bar_plots(datasets, x_label):
    # Define colors for different class weights
    colors = plt.cm.viridis(np.linspace(0, 1, len(class_weights_list)))

    # 1. Plot DP Difference between Teacher and Student
    plt.figure(figsize=(12, 7))
    bar_width = 0.15
    index = np.arange(len(datasets))

    for i, class_weights in enumerate(class_weights_list):
        avg_diffs = [auc_diffs[dataset][class_weights]['avg_diffs_dp_group'] for dataset in datasets]
        std_diffs = [auc_diffs[dataset][class_weights]['std_diffs_dp_group'] for dataset in datasets]

        plt.bar(index + i*bar_width - bar_width*2, avg_diffs, bar_width, 
                yerr=np.array(std_diffs)/np.sqrt(len(seeds)), 
                capsize=3, 
                label=f'Class Weights: {class_weights}',
                color=colors[i])

    plt.title('Group 0-1 DP Difference (Teacher-Student) vs ' + x_label)
    plt.xlabel(x_label)
    plt.ylabel('Group 0-1 DP Difference (Teacher-Student)')
    plt.xticks(index, datasets)
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f'{figs_folder}/group_0_1_dp_diff_vs_{x_label[-2:-1]}_all_weights.png')
    plt.close()

    # 2. Plot DP Comparison for Teacher and Student
    plt.figure(figsize=(15, 10))
    bar_width = 0.08
    index = np.arange(len(datasets))

    for i, class_weights in enumerate(class_weights_list):
        avg_teacher = [auc_diffs[dataset][class_weights]['avg_teachers_dp_group'] for dataset in datasets]
        std_teacher = [auc_diffs[dataset][class_weights]['std_teachers_dp_group'] for dataset in datasets]

        avg_student = [auc_diffs[dataset][class_weights]['avg_students_dp_group'] for dataset in datasets]
        std_student = [auc_diffs[dataset][class_weights]['std_students_dp_group'] for dataset in datasets]

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

    plt.title('Group 0-1 DP Comparison vs '+ x_label)
    plt.xlabel(x_label)
    plt.ylabel('Group 0-1 DP')
    plt.xticks(index, datasets)
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f'{figs_folder}/group_0_1_dp_comparison_vs_{x_label[-2:-1]}_all_weights.png')
    plt.close()

    # 3. Plot Overall DP Difference
    plt.figure(figsize=(12, 7))
    bar_width = 0.15
    index = np.arange(len(datasets))

    for i, class_weights in enumerate(class_weights_list):
        avg_diffs = [auc_diffs[dataset][class_weights]['avg_diffs_dp'] for dataset in datasets]
        std_diffs = [auc_diffs[dataset][class_weights]['std_diffs_dp'] for dataset in datasets]

        plt.bar(index + i*bar_width - bar_width*2, avg_diffs, bar_width, 
                yerr=np.array(std_diffs)/np.sqrt(len(seeds)), 
                capsize=3, 
                label=f'Class Weights: {class_weights}',
                color=colors[i])

    plt.title('Overall DP Difference (Teacher-Student) vs '+ x_label)
    plt.xlabel(x_label)
    plt.ylabel('Overall DP Difference (Teacher-Student)')
    plt.xticks(index, datasets)
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f'{figs_folder}/overall_dp_diff_vs_{x_label[-2:-1]}_all_weights.png')
    plt.close()
    
    # 4. Plot EO Difference between Teacher and Student
    plt.figure(figsize=(12, 7))
    bar_width = 0.15
    index = np.arange(len(datasets))

    for i, class_weights in enumerate(class_weights_list):
        avg_diffs = [auc_diffs[dataset][class_weights]['avg_diffs_eo_group'] for dataset in datasets]
        std_diffs = [auc_diffs[dataset][class_weights]['std_diffs_eo_group'] for dataset in datasets]

        plt.bar(index + i*bar_width - bar_width*2, avg_diffs, bar_width, 
                yerr=np.array(std_diffs)/np.sqrt(len(seeds)), 
                capsize=3, 
                label=f'Class Weights: {class_weights}',
                color=colors[i])

    plt.title('Group 0-1 EO Difference (Teacher-Student) vs ' + x_label)
    plt.xlabel(x_label)
    plt.ylabel('Group 0-1 EO Difference (Teacher-Student)')
    plt.xticks(index, datasets)
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f'{figs_folder}/group_0_1_eo_diff_vs_{x_label[-2:-1]}_all_weights.png')
    plt.close()

    # 5. Plot EO Comparison for Teacher and Student
    plt.figure(figsize=(15, 10))
    bar_width = 0.08
    index = np.arange(len(datasets))

    for i, class_weights in enumerate(class_weights_list):
        avg_teacher = [auc_diffs[dataset][class_weights]['avg_teachers_eo_group'] for dataset in datasets]
        std_teacher = [auc_diffs[dataset][class_weights]['std_teachers_eo_group'] for dataset in datasets]

        avg_student = [auc_diffs[dataset][class_weights]['avg_students_eo_group'] for dataset in datasets]
        std_student = [auc_diffs[dataset][class_weights]['std_students_eo_group'] for dataset in datasets]

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

    plt.title('Group 0-1 EO Comparison vs '+ x_label)
    plt.xlabel(x_label)
    plt.ylabel('Group 0-1 EO')
    plt.xticks(index, datasets)
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f'{figs_folder}/group_0_1_eo_comparison_vs_{x_label[-2:-1]}_all_weights.png')
    plt.close()

    # 6. Plot Overall EO Difference
    plt.figure(figsize=(12, 7))
    bar_width = 0.15
    index = np.arange(len(datasets))

    for i, class_weights in enumerate(class_weights_list):
        avg_diffs = [auc_diffs[dataset][class_weights]['avg_diffs_eo'] for dataset in datasets]
        std_diffs = [auc_diffs[dataset][class_weights]['std_diffs_eo'] for dataset in datasets]

        plt.bar(index + i*bar_width - bar_width*2, avg_diffs, bar_width, 
                yerr=np.array(std_diffs)/np.sqrt(len(seeds)), 
                capsize=3, 
                label=f'Class Weights: {class_weights}',
                color=colors[i])

    plt.title('Overall EO Difference (Teacher-Student) vs '+ x_label)
    plt.xlabel(x_label)
    plt.ylabel('Overall EO Difference (Teacher-Student)')
    plt.xticks(index, datasets)
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f'{figs_folder}/overall_eo_diff_vs_{x_label[-2:-1]}_all_weights.png')
    plt.close()


def create_metric_heatmap(metric_key, title, colorbar_label, datasets, x_label, filename_suffix):
    matrix = np.zeros((len(class_weights_list), len(datasets)))

    for i, class_weights in enumerate(class_weights_list):
        for j, dataset in enumerate(datasets):
            matrix[i, j] = auc_diffs[dataset][class_weights][metric_key]

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0,
        xticklabels=datasets,
        yticklabels=class_weights_list,
        cbar_kws={'label': colorbar_label}
    )
    plt.title(title + ' by Class Weight and ' + x_label)
    plt.xlabel(x_label)
    plt.ylabel('Class Weights')
    plt.tight_layout()
    plt.savefig(f'{figs_folder}/heatmap_{filename_suffix}_vs_{x_label}.png')
    plt.close()


# +
def create_line_plots(datasets, x_label, metric_name, metric_label=None, title_prefix=None, filename_suffix=None):
    # Set global min/max if available, otherwise auto-scale
#     vmin, vmax = global_limits.get(metric_name, (None, None))

    colors = plt.cm.viridis(np.linspace(0, 1, len(class_weights_list)))
    plt.figure(figsize=(15, 10))
    x_positions = np.arange(len(datasets))

    for i, class_weights in enumerate(class_weights_list):
        avg_teacher = [auc_diffs[dataset][class_weights][f'avg_teachers_{metric_name}'] for dataset in datasets]
        std_teacher = [auc_diffs[dataset][class_weights][f'std_teachers_{metric_name}'] for dataset in datasets]

        avg_student = [auc_diffs[dataset][class_weights][f'avg_students_{metric_name}'] for dataset in datasets]
        std_student = [auc_diffs[dataset][class_weights][f'std_students_{metric_name}'] for dataset in datasets]

        plt.plot(x_positions, avg_teacher, '-o',
                 label=f'Teacher {class_weights}',
                 color=colors[i],
                 linewidth=2,
                 markersize=8)

        plt.fill_between(x_positions,
                         np.array(avg_teacher) - np.array(std_teacher)/np.sqrt(len(seeds)),
                         np.array(avg_teacher) + np.array(std_teacher)/np.sqrt(len(seeds)),
                         color=colors[i], alpha=0.2)

        plt.plot(x_positions, avg_student, '--s',
                 label=f'Student {class_weights}',
                 color=colors[i],
                 linewidth=2,
                 markersize=8)

        plt.fill_between(x_positions,
                         np.array(avg_student) - np.array(std_student)/np.sqrt(len(seeds)),
                         np.array(avg_student) + np.array(std_student)/np.sqrt(len(seeds)),
                         color=colors[i], alpha=0.1)

#     if vmin is not None and vmax is not None:
#         plt.ylim(vmin, vmax)

    # Generate labels
    metric_label = metric_label or metric_name.replace('_', ' ').title()
    title_prefix = title_prefix or ''
    filename_suffix = filename_suffix or metric_name

    plt.title(f'{title_prefix}{metric_label} vs {x_label}')
    plt.xlabel(x_label)
    plt.ylabel(metric_label)
    plt.xticks(x_positions, datasets)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f'{figs_folder}/{metric_name}_vs_{x_label}_line.png')
    plt.close()


# +
# # First, determine global min and max values across all datasets
# def find_global_min_max(datasets_list):
#     global_min_diff = float('inf')
#     global_max_diff = float('-inf')
#     global_min_auc = float('inf')
#     global_max_auc = float('-inf')
#     global_min_acc = float('inf')
#     global_max_acc = float('-inf')
#     global_min_auc_teacher_student = float('inf')
#     global_max_auc_teacher_student = float('-inf')
    
#     for datasets_type in datasets_list:
#         for dataset in datasets_type:
#             for class_weight in class_weights_list:
#                 # For group differences
#                 diff_val = auc_diffs[dataset][class_weight]['avg_diff']
#                 global_min_diff = min(global_min_diff, diff_val)
#                 global_max_diff = max(global_max_diff, diff_val)
                
#                 # For overall AUC values
#                 auc_val = auc_diffs[dataset][class_weight]['avg_aucs']
#                 global_min_auc = min(global_min_auc, auc_val)
#                 global_max_auc = max(global_max_auc, auc_val)
                
#                 # For teacher/student AUC values
#                 avg_teacher = auc_diffs[dataset][class_weight]['avg_teacher']
#                 avg_student = auc_diffs[dataset][class_weight]['avg_student']

#                 # Update global min and max
#                 global_min_auc_teacher_student = min(global_min_auc_teacher_student, avg_teacher, avg_student)
#                 global_max_auc_teacher_student = max(global_max_auc_teacher_student, avg_teacher, avg_student)

#                 # For Accuracy values
#                 acc_val = auc_diffs[dataset][class_weight]['avg_accs']
#                 global_min_acc = min(global_min_acc, acc_val)
#                 global_max_acc = max(global_max_acc, acc_val)
    
#     # Add some padding to the limits (e.g., 5%)
#     padding_diff = (global_max_diff - global_min_diff) * 0.05
#     padding_auc = (global_max_auc - global_min_auc) * 0.05
#     padding_acc = (global_max_acc - global_min_acc) * 0.05
#     padding_auc_ts = (global_max_auc_teacher_student - global_min_auc_teacher_student) * 0.05
    
#     return {
#         'diff': (global_min_diff - padding_diff, global_max_diff + padding_diff),
#         'auc': (global_min_auc - padding_auc, global_max_auc + padding_auc),
#         'acc': (global_min_acc - padding_acc, global_max_acc + padding_acc),
#         'auc_ts': (global_min_auc_teacher_student - padding_auc_ts, global_max_auc_teacher_student + padding_auc_ts)
        
#     }

# # Calculate global limits before creating any plots
# global_limits = find_global_min_max([datasets_p, datasets_q])

# +
for (dataset, x_label) in [(datasets_p, 'Group Balance (p)'),(datasets_q, 'Edge probability Ratio (q)')]:
#     create_bar_plots(dataset, x_label)
    
#     create_metric_heatmap(
#         'avg_diffs_dp_group',
#         'Group 0-1 DP Difference (T - S)',
#         'DP Diff',
#         dataset,
#         x_label,
#         'dp_diff_group'
#     )
    
#     create_metric_heatmap(
#         'avg_diffs_eo_group',
#         'Group 0-1 EO Difference (T - S)',
#         'DP Diff',
#         dataset,
#         x_label,
#         'eo_diff_group'
#     )
    
#     create_metric_heatmap(
#         'avg_diffs_dp',
#         'DP Difference (T - S)',
#         'DP Diff',
#         dataset,
#         x_label,
#         'dp_diff'
#     )
    
#     create_metric_heatmap(
#         'avg_diffs_eo',
#         'EO Difference (T - S)',
#         'EO Diff',
#         dataset,
#         x_label,
#         'eo_diff'
#     )
    
    create_line_plots(
        datasets=dataset,
        x_label=x_label,
        metric_name='dp_group',
        metric_label='Group 0-1 DP',
        title_prefix='Group 0-1 ',
        filename_suffix='dp_diff_group'
    )
    
    create_line_plots(
        datasets=dataset,
        x_label=x_label,
        metric_name='dp',
        metric_label='DP',
        title_prefix='',
        filename_suffix='dp_diff'
    )
    
    create_line_plots(
        datasets=dataset,
        x_label=x_label,
        metric_name='eo_group',
        metric_label='Group 0-1 EO',
        title_prefix='Group 0-1 ',
        filename_suffix='eo_diff_group'
    )
    
    create_line_plots(
        datasets=dataset,
        x_label=x_label,
        metric_name='eo',
        metric_label='EO',
        title_prefix='',
        filename_suffix='eo_diff'
    )


