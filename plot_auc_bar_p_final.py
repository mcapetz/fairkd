import subprocess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

datasets = ["0.1", "0.2", "0.3", "0.4", "0.5"] # these are the p values
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
            teacher_val = np.load(f'saved_arrays/auc_ovr_diff_teacher_sbm{dataset}_{seed}_c={class_weights}_p.npy')
            student_val = np.load(f'saved_arrays/auc_ovr_diff_student_sbm{dataset}_{seed}_c={class_weights}_p.npy')
            teacher_auc = np.load(f'saved_arrays/auc_ovr_overall_teacher_sbm{dataset}_{seed}_c={class_weights}_p.npy')
            student_auc = np.load(f'saved_arrays/auc_ovr_overall_student_sbm{dataset}_{seed}_c={class_weights}_p.npy')
            teacher_acc = np.load(f'saved_arrays/acc_teacher_sbm{dataset}_{seed}_c={class_weights}_p.npy')
            student_acc = np.load(f'saved_arrays/acc_student_sbm{dataset}_{seed}_c={class_weights}_p.npy')
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

# Extracting data for plotting
for class_weights in class_weights_list:
    
    categories = list(auc_diffs.keys())
    avg_diffs = [auc_diffs[cat][class_weights]['avg_diff'] for cat in categories]
    std_diffs = [auc_diffs[cat][class_weights]['std_diff'] for cat in categories]

    # Creating the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(categories, avg_diffs, yerr=std_diffs/np.sqrt(len(seeds)), capsize=5)
    plt.title('Group 0 - 1 AUC Diff between Teacher and Student vs Group Balance')
    plt.xlabel('Group Balance')
    plt.ylabel('Difference between Teacher and Student')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'figs_final/auc_bar_diff_c={class_weights}_p.png')
    plt.close() 

    # Extracting data for plotting
    categories = list(auc_diffs.keys())
    avg_teacher = [auc_diffs[cat][class_weights]['avg_teacher'] for cat in categories]
    std_teacher = [auc_diffs[cat][class_weights]['std_teacher'] for cat in categories]
    avg_student = [auc_diffs[cat][class_weights]['avg_student'] for cat in categories]
    std_student = [auc_diffs[cat][class_weights]['std_student'] for cat in categories]

    # Creating the bar plot
    bar_width = 0.35
    index = range(len(categories))
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # with error bars
    bar1 = ax.bar(index, avg_teacher, bar_width, label='Average Teacher', yerr=std_teacher/np.sqrt(len(seeds)), capsize=5)
    bar2 = ax.bar([i + bar_width for i in index], avg_student, bar_width, label='Average Student', yerr=std_student/np.sqrt(len(seeds)), capsize=5)

    ax.set_title('Group 0 - 1 AUC Comparison vs Group Balance')
    ax.set_xlabel('Group Balance')
    ax.set_ylabel('Group 0 - 1 AUC')
    ax.set_xticks([i + bar_width/2 for i in index])
    ax.set_xticklabels(categories, rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'figs_final/auc_bar_comp_c={class_weights}_p.png')
    plt.close() 


    # Extracting data for plotting
    categories = list(auc_diffs.keys())
    avg_diffs = [auc_diffs[cat][class_weights]['avg_aucs'] for cat in categories]
    std_diffs = [auc_diffs[cat][class_weights]['std_aucs'] for cat in categories]

    # Creating the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(categories, avg_diffs, yerr=std_diffs/np.sqrt(len(seeds)), capsize=5)
    plt.title('Overall AUC Diff (Teacher-Student) vs Group Balance')
    plt.xlabel('Group Balance')
    plt.ylabel('Overall AUC Diff (Teacher-Student)')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f'figs_final/auc_c={class_weights}_p.png')
    plt.close() 

    # Extracting data for plotting
    categories = list(auc_diffs.keys())
    avg_diffs = [auc_diffs[cat][class_weights]['avg_accs'] for cat in categories]
    std_diffs = [auc_diffs[cat][class_weights]['std_accs'] for cat in categories]

    # Creating the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(categories, avg_diffs, yerr=std_diffs/np.sqrt(len(seeds)), capsize=5)
    plt.title('Overall Accuracy Diff (Teacher-Student) vs Group Balance')
    plt.xlabel('Group Balance')
    plt.ylabel('Overall Accuracy Diff (Teacher-Student)')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f'figs_final/acc_c={class_weights}_p.png')
    plt.close()
