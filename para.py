import sys, os , shutil
import numpy as np

para_range = [0.000005, 0.005]
para_test_num = 18

#use log scale to sample the parameter
log_para_range = [np.log10(para_range[0]), np.log10(para_range[1])]
log_para_samples = np.linspace(log_para_range[0], log_para_range[1], para_test_num)
para_samples = [10**x for x in log_para_samples]

for para in para_samples:
    cmd = f'CUDA_VISIBLE_DEVICES=6 python train.py -s data/DTU/scan24/ -m output/test1.5_{para} --port 7645 --replace_params_name app_opc_opc_loss --replace_params_value {para}' 
    print(cmd)
    os.system(cmd)



#1.5scaling_lr
#2.5feature_lr #0.0025