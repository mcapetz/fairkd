# -*- coding: utf-8 -*-
# +
def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(16)
# -

from os import environ
import argparse
import subprocess
import random
import numpy as np
import seaborn as sns
import os

# Settings
num_runs = 5
datasets = ["sbm0.1", "sbm0.2"] #, "sbm0.3", "sbm0.4", "sbm0.5", "sbm0.05", "sbm0.25", "sbm0.50", "sbm0.75", "sbm1.00"
class_weights_list = ["0.1,0.9", "0.2,0.8", "0.3,0.7", "0.4,0.6", "0.5,0.5"]

failed_runs = []

for i in range(num_runs):    
    for class_weights in class_weights_list:
        print("class_weights: ", class_weights)
        environ['class_weights'] = class_weights
    
        for dataset in datasets:
            # Construct the filename
            filename = f"acc_teacher_{dataset}_{i}_c={class_weights}.npy"
            file_path = os.path.join("saved_arrays", filename)
            
            # check if it is already done
            if os.path.exists(file_path): 
                print("\t", dataset, class_weights, "already done")
                continue
            
            print("\t", "dataset: ", dataset)
            print("\t", "teacher")
            script_to_run = 'train_teacher.py'
            script_arguments = ['--exp_setting', 'tran', '--teacher', 'GCN', '--dataset', dataset, '--seed', str(i)]
            try:
                result = subprocess.run(['python', script_to_run] + script_arguments, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                print("\t", "\t", dataset, class_weights,"error")
                num_failures = e.stdout.count("✗")
                entry_line = f"{dataset}_{class_weights}_num_failures:{num_failures}\n"
                
                # Check if the line already exists
                already_logged = False
                if os.path.exists("error_log_1.txt"):
                    with open("error_log_1.txt", "r") as f:
                        for line in f:
                            if line.startswith(f"{dataset}_{class_weights}_num_failures:"):
                                already_logged = True
                                break

                # Only write if not already logged
                if not already_logged:
                    with open("error_log_1.txt", "a") as f:
                        f.write(entry_line)
                        f.write(f"{e.stdout}\n{e.stderr}")
                        
                continue # skip student if teacher fails
            
            print("student")
            script_to_run = 'train_student.py'
            script_arguments = ['--exp_setting', 'tran', '--teacher', 'GCN', '--student', 'MLP', '--dataset', dataset, '--out_t_path', 'outputs', '--seed', str(i)]
            try:
                result = subprocess.run(['python', script_to_run] + script_arguments, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                print("\t", "\t", dataset, class_weights,"error")
                num_failures = e.stdout.count("✗")
                entry_line = f"{dataset}_{class_weights}_num_failures:{num_failures}\n"
                
                # Check if the line already exists
                already_logged = False
                if os.path.exists("error_log_1.txt"):
                    with open("error_log_1.txt", "r") as f:
                        for line in f:
                            if line.startswith(f"{dataset}_{class_weights}_num_failures:"):
                                already_logged = True
                                break

                # Only write if not already logged
                if not already_logged:
                    with open("error_log_1.txt", "a") as f:
                        f.write(entry_line)
                        f.write(f"{e.stdout}\n{e.stderr}")

