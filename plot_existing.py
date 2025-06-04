# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from scipy import stats as scipy_stats
import matplotlib.cm as cm

seeds = ["0", "1", "2", "3", "4"]

figs_folder = "figs2"
saved_arrays_folder = "saved_arrays"
datasets = ["German", "Credit", "NBA", "syn-1", "syn-2", "sport", "occupation"]

# +
# load in the auc values

for dataset in datasets:
    
    auc_dict[dataset] = dict()
    
    teacher_auc_diffs = []
    student_auc_diffs = []
    auc_diffs = []
    
    teacher_acc_diffs = []
    student_acc_diffs = []
    acc_diffs = []
    
    # avg and std for seeds
    for seed in seeds: 
        teacher_auc_diff = abs(np.load(f'{saved_arrays_folder}/auc_ovr_diff_teacher_{seed}_{dataset}.npy'))
        student_auc_diff = abs(np.load(f'{saved_arrays_folder}/auc_ovr_diff_student_{seed}_{dataset}.npy'))
        auc_diff = teacher_auc_diff - student_auc_diff

        dp_0, dp_1, eo_0, eo_1, dp, eo, teacher_acc_diff = np.load(f'{saved_arrays_folder}/trad_metrics_teacher_{seed}_{dataset}.npy')
        dp_0_, dp_1_, eo_0_, eo_1_, dp_, eo_, student_acc_diff = np.load(f'{saved_arrays_folder}/trad_metrics_student_{seed}_{dataset}.npy')
        teacher_acc_diff = abs(teacher_acc_diff)
        student_acc_diff = abs(student_acc_diff)
        acc_diff = teacher_acc_diff - student_acc_diff

        teacher_auc_diffs.append(teacher_auc_diff)
        student_auc_diffs.append(student_auc_diff)
        auc_diffs.append(auc_diff)

        teacher_acc_diffs.append(teacher_acc_diff)
        student_acc_diffs.append(student_acc_diff)
        acc_diffs.append(acc_diff)


    auc_dict[dataset] = {
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
