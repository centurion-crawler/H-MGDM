import os
import torch
import numpy as np
import torch.nn.functional as F
from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_censored

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_info(file_name, info):
    with open(file_name, "a") as f:
        f.write(info)


def adjust_learning_rate(optimizer, lr, epoch, lr_step, lr_gamma):
    lr = lr * (lr_gamma ** (epoch // lr_step)) 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  


def calculate_ci(patient_surv_time, patient_hazards, patient_censorship):
    surv_time_list, hazards_list, censorship_list = [], [], []
    for patient_id in patient_hazards:
        surv_time_list.append(patient_surv_time[patient_id])
        hazards_list.append(-1 * patient_hazards[patient_id])
        censorship_list.append(patient_censorship[patient_id])
    c_index = concordance_index(surv_time_list, hazards_list, censorship_list)
    return c_index


def calculate_all_ci(cli_dict, patient_hazards):
    surv_time_list, hazards_list, censorship_list = [], [], []
    for patient_id in patient_hazards:
        surv_time_list.append(cli_dict[patient_id][1])
        hazards_list.append(-1 * patient_hazards[patient_id])
        censorship_list.append(cli_dict[patient_id][0])
    c_index = concordance_index(surv_time_list, hazards_list, censorship_list)
    return c_index   


def CoxSurvLoss(hazards, time, censorship):
    current_batch_len = len(hazards)
    R_matrix_train = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_matrix_train[i, j] = time[j] >= time[i]

    train_R = torch.FloatTensor(R_matrix_train)
    train_R = train_R.cuda()
    train_ystatus = torch.tensor(np.array(censorship), dtype=torch.float).to(device)
    theta = hazards.reshape(-1)

    exp_theta = torch.exp(theta)
    loss = - torch.mean((theta - torch.log(torch.sum(exp_theta * train_R, dim=1))) * train_ystatus)
    return loss 


def DC_Loss(feats_x, feats_y):
    matrix_a = torch.sqrt(torch.sum(torch.square(feats_x.unsqueeze(0) - feats_x.unsqueeze(1)), dim=-1) + 1e-12)
    matrix_b = torch.sqrt(torch.sum(torch.square(feats_y.unsqueeze(0) - feats_y.unsqueeze(1)), dim=-1) + 1e-12)

    matrix_A = matrix_a - torch.mean(matrix_a, dim=0, keepdims=True) - torch.mean(matrix_a, dim=1, keepdims=True) + torch.mean(matrix_a)
    matrix_B = matrix_b - torch.mean(matrix_b, dim=0, keepdims=True) - torch.mean(matrix_b, dim=1, keepdims=True) + torch.mean(matrix_b)

    Gamma_XY = torch.sum(matrix_A * matrix_B) / (matrix_A.shape[0] * matrix_A.shape[1])
    Gamma_XX = torch.sum(matrix_A * matrix_A) / (matrix_A.shape[0] * matrix_A.shape[1])
    Gamma_YY = torch.sum(matrix_B * matrix_B) / (matrix_A.shape[0] * matrix_A.shape[1])

    dc_loss = Gamma_XY / torch.sqrt(Gamma_XX * Gamma_YY + 1e-9)
    return dc_loss


def Fusion_Loss(x_slide, x_omic, x_general, sigma2_slide, sigma2_omic):
    dis_slide = torch.norm(x_general - x_slide, dim=1, keepdim=True)
    slide_loss = dis_slide / sigma2_slide + torch.log(sigma2_slide)
    slide_loss = torch.mean(slide_loss)
    
    dis_omic = torch.norm(x_general - x_omic, dim=1, keepdim=True)
    omic_loss = dis_omic / sigma2_omic + torch.log(sigma2_omic)
    omic_loss = torch.mean(omic_loss)
    
    return slide_loss + omic_loss
