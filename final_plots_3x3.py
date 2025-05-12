import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

datasets_p = ["0.1", "0.2", "0.3", "0.4", "0.5"] # these are the p values
datasets_q = ["0.05", "0.25", "0.50", "0.75", "1.00"] # these are the p values
datasets = datasets_p + datasets_q
seeds = ["0", "1", "2", "3", "4"]
class_weights_list = ["0.1,0.9", "0.2,0.8", "0.3,0.7", "0.4,0.6", "0.5,0.5"]

auc_dict = dict()

figs_folder = "figs_3x3"
saved_arrays_folder = "saved_arrays"

# load in the auc values
for dataset in datasets:  
    
    auc_dict[dataset] = dict()
    
    teacher_auc_diffs = []
    student_auc_diffs = []
    auc_diffs = []
    
    teacher_acc_diffs = []
    student_acc_diffs = []
    acc_diffs = []

    
    for class_weights in class_weights_list:

        # avg and std for seeds
        for seed in seeds: 
            teacher_auc_diff = abs(np.load(f'{saved_arrays_folder}/auc_ovr_diff_teacher_sbm{dataset}_{seed}_c={class_weights}.npy'))
            student_auc_diff = abs(np.load(f'{saved_arrays_folder}/auc_ovr_diff_student_sbm{dataset}_{seed}_c={class_weights}.npy'))
            auc_diff = teacher_auc_diff - student_auc_diff
            
            dp_0, dp_1, eo_0, eo_1, dp, eo, teacher_acc_diff = abs(np.load(f'{saved_arrays_folder}/trad_metrics_teacher_sbm{dataset}_{seed}_c={class_weights}.npy'))
            dp_0_, dp_1_, eo_0_, eo_1_, dp_, eo_, student_acc_diff = abs(np.load(f'{saved_arrays_folder}/trad_metrics_student_sbm{dataset}_{seed}_c={class_weights}.npy'))
            acc_diff = teacher_acc_diff - student_acc_diff

            teacher_auc_diffs.append(teacher_auc_diff)
            student_auc_diffs.append(student_auc_diff)
            auc_diffs.append(auc_diff)
            
            teacher_acc_diffs.append(teacher_acc_diff)
            student_acc_diffs.append(student_acc_diff)
            acc_diffs.append(acc_diff)
            
        
        auc_dict[dataset][class_weights] = {
            "avg_teacher_auc_diffs": np.mean(teacher_auc_diffs, axis=0),
            "std_teacher_auc_diffs": np.std(teacher_auc_diffs, axis=0),
            "avg_student_auc_diffs": np.mean(student_auc_diffs, axis=0),
            "std_student_auc_diffs": np.std(student_auc_diffs, axis=0),
            "avg_auc_diff": np.mean(auc_diffs, axis=0),
            "std_auc_diff": np.std(auc_diffs, axis=0),
            "avg_teacher_acc_diffs": np.mean(teacher_acc_diffs, axis=0),
            "std_teacher_acc_diffs": np.std(teacher_acc_diffs, axis=0),
            "avg_student_acc_diffs": np.mean(student_acc_diffs, axis=0),
            "std_student_acc_diffs": np.std(student_acc_diffs, axis=0),
            "avg_acc_diff": np.mean(acc_diffs, axis=0),
            "std_acc_diff": np.std(acc_diffs, axis=0),
            
        }


def create_heatmap_grid(auc_dict, datasets_p, datasets_q, class_weights_list, 
                        metric_type='acc', save_path=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, wspace=0.3, hspace=0.4)

    row_pairs = [
        ('class_balance', 'group_balance'),
        ('class_balance', 'edge_ratio'),
        ('group_balance', 'edge_ratio')
    ]
    col_metrics = ['teacher', 'student', 'diff']
    col_titles = ['Teacher Fairness', 'Student Fairness', 'Teacher - Student Fairness']

    # Precompute common vmin/vmax for teacher and student metrics
    teacher_vals = []
    student_vals = []
    teacher_student_vals = []

    for p in datasets_p:
        for cw in class_weights_list:
            try:
                teacher_vals.append(auc_dict[p][cw][f"avg_teacher_{metric_type}_diffs"])
                student_vals.append(auc_dict[p][cw][f"avg_student_{metric_type}_diffs"])
                teacher_student_vals.append(auc_dict[p][cw][f"avg_teacher_{metric_type}_diffs"] - auc_dict[p][cw][f"avg_student_{metric_type}_diffs"])
            except KeyError:
                continue
    for q in datasets_q:
        for cw in class_weights_list:
            try:
                teacher_vals.append(auc_dict[q][cw][f"avg_teacher_{metric_type}_diffs"])
                student_vals.append(auc_dict[q][cw][f"avg_student_{metric_type}_diffs"])
                teacher_student_vals.append(auc_dict[q][cw][f"avg_teacher_{metric_type}_diffs"] - auc_dict[q][cw][f"avg_student_{metric_type}_diffs"])
            except KeyError:
                continue

    shared_vmin = min(teacher_vals + student_vals)
    shared_vmax = max(teacher_vals + student_vals)
    
    teacher_student_vmin = min(teacher_student_vals)
    teacher_student_vmax = max(teacher_student_vals)

    
    # Now draw the heatmaps with consistent color scale
    for row_idx, (factor1, factor2) in enumerate(row_pairs):
        for col_idx, metric in enumerate(col_metrics):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            if row_idx == 0:
                x_values = datasets_p
                y_values = class_weights_list
                x_label = 'Group Balance (p)'
                y_label = 'Class Balance'
                data_matrix = np.full((len(y_values), len(x_values)), np.nan)
                for i, class_weight in enumerate(y_values):
                    for j, p_val in enumerate(x_values):
                        key = f"avg_teacher_{metric_type}_diffs" if metric == 'teacher' else (
                            f"avg_student_{metric_type}_diffs" if metric == 'student' else f"avg_{metric_type}_diff")
                        try:
                            data_matrix[i, j] = auc_dict[p_val][class_weight][key]
                        except KeyError:
                            pass

            elif row_idx == 1:
                x_values = datasets_q
                y_values = class_weights_list
                x_label = 'Edge Ratio (q)'
                y_label = 'Class Balance'
                data_matrix = np.full((len(y_values), len(x_values)), np.nan)
                for i, class_weight in enumerate(y_values):
                    for j, q_val in enumerate(x_values):
                        key = f"avg_teacher_{metric_type}_diffs" if metric == 'teacher' else (
                            f"avg_student_{metric_type}_diffs" if metric == 'student' else f"avg_{metric_type}_diff")
                        try:
                            data_matrix[i, j] = auc_dict[q_val][class_weight][key]
                        except KeyError:
                            pass

            else:
                x_values = datasets_q
                y_values = datasets_p
                x_label = 'Edge Ratio (q)'
                y_label = 'Group Balance (p)'
                fixed_class_weight = class_weights_list[2]
                data_matrix = np.full((len(y_values), len(x_values)), np.nan)
                for i, p_val in enumerate(y_values):
                    for j, q_val in enumerate(x_values):
                        try:
                            if metric == 'teacher':
                                val = (auc_dict[q_val][fixed_class_weight][f"avg_teacher_{metric_type}_diffs"] +
                                       auc_dict[p_val][fixed_class_weight][f"avg_teacher_{metric_type}_diffs"]) / 2
                            elif metric == 'student':
                                val = (auc_dict[q_val][fixed_class_weight][f"avg_student_{metric_type}_diffs"] +
                                       auc_dict[p_val][fixed_class_weight][f"avg_student_{metric_type}_diffs"]) / 2
                            else:
                                val = (auc_dict[q_val][fixed_class_weight][f"avg_{metric_type}_diff"] +
                                       auc_dict[p_val][fixed_class_weight][f"avg_{metric_type}_diff"]) / 2
                            data_matrix[i, j] = val
                        except KeyError:
                            pass

            
            # Set vmin and vmax based on metric
            if metric in ['teacher', 'student']:
                vmin = shared_vmin
                vmax = shared_vmax
            else:  # diff heatmap can have its own scale
                vmin = teacher_student_vmin
                vmax = teacher_student_vmax

            sns.heatmap(
                data_matrix, annot=True, cmap='viridis', ax=ax, fmt='.3f',
                xticklabels=x_values, yticklabels=[str(cw)[:3] for cw in y_values], 
                cbar_kws={'label': 'Difference'},
                vmin=vmin, vmax=vmax
            )

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(col_titles[col_idx])

    plt.suptitle('Accuracy Fairness Differences (0-1)' if metric_type == 'acc' 
                 else 'AUC Fairness Differences (0-1)', fontsize=16, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return fig




# +
# Create main figure with accuracy differences
acc_fig = create_heatmap_grid(
    auc_dict=auc_dict,
    datasets_p=datasets_p,
    datasets_q=datasets_q,
    class_weights_list=class_weights_list,
    metric_type='acc',
    save_path=f"{figs_folder}/fairness_heatmaps_acc.png"
)

# Create appendix figure with AUC differences
auc_fig = create_heatmap_grid(
    auc_dict=auc_dict,
    datasets_p=datasets_p,
    datasets_q=datasets_q,
    class_weights_list=class_weights_list,
    metric_type='auc',
    save_path=f"{figs_folder}/fairness_heatmaps_auc.png"
)
