import os
import argparse
def get_params():
    parser = argparse.ArgumentParser(description="All the parameters of this network.")
    parser.add_argument("--cancer_type", type=str, default="KIRC_KimiaNet", help="cancer_type")
    parser.add_argument("--model_name", type=str, default="AMIL", help="choose model to train")
    parser.add_argument("--details", type=str, default="", help="details")
    parser.add_argument("--slide_in_feats", type=int, default=1024, help="slide_in_feats")
    parser.add_argument("--fusion_dim", type=int, default=128, help="fusion_dim")
    parser.add_argument("--output_dim", type=int, default=1, help="output_dim")
    parser.add_argument("--slide_size_arg", type=str, choices=["small","big"], default="small", help="slide_size_arg")
    # parser.add_argument("--omic_size_arg", type=str, choices=["small","big"], default="small", help="omic_size_arg")
    parser.add_argument("--lr", type=float, default=1e-4, help="lr")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--epochs", type=int, default=100, help="epochs")
    parser.add_argument("--folds_num", type=int, default=5, help="folds_num")
    parser.add_argument("--batch_size", type=int, default=16, help="batch_size")
    parser.add_argument("--if_adjust_lr", action="store_true", default=True, help="if_adjust_lr")
    parser.add_argument("--dropout", type=float, default=0.25, help="dropout")
    parser.add_argument("--dc_loss_ratio", type=float, default=0.05, help="dc_loss_ratio")
    parser.add_argument("--fusion_loss_ratio", type=float, default=0.005, help="fusion_loss_ratio")
    parser.add_argument("--result_dir", type=str, default="./results", help="result_dir")
    parser.add_argument("--gpu", type=str, default="5", help="gpu")
    args, _ = parser.parse_known_args()
    return args
args = get_params()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import copy
import torch
import joblib
import random
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm 
from utils import *
from model import AMIL_Surv, PatchGCN_Surv, DeepGraphConv_Surv

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def setup_seed(seed):
	import random
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if device.type == 'cuda':
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True



def train_one_epoch(model,patients,slide_feats_dict,cli_dict,optimizer,args):
    train_batch_hazards = []
    train_batch_surv_time = []
    train_batch_censorship = []


    train_all_loss = 0.
    train_patients_hazards = {}
    train_patients_surv_time = {}
    train_patients_censorship = {}
    
    model.train()
    random.shuffle(patients)
    for n_iter, patient_id in tqdm(enumerate(patients)):
        surv_time = cli_dict[patient_id][1]
        censorship = cli_dict[patient_id][0]
        train_patients_surv_time[patient_id] = surv_time
        train_patients_censorship[patient_id] = censorship
        train_batch_censorship.append(censorship)
        train_batch_surv_time.append(surv_time)
        if args.model_name in ['PatchGCN','DeepGraphConv']:
            slide_feats_dict[patient_id].x = slide_feats_dict[patient_id].x[:,:1024]
        else:
            slide_feats_dict[patient_id] = slide_feats_dict[patient_id][:,:1024]
        slide_data = slide_feats_dict[patient_id].to(device)
        hazards = model(slide_data)

        hazards = hazards.squeeze(0)
        train_batch_hazards.append(hazards)
        train_patients_hazards[patient_id] = hazards.detach().cpu().numpy()
        
        if ((n_iter+1)%args.batch_size == 0 and (n_iter+1)//args.batch_size != len(patients)//args.batch_size) or n_iter == len(patients) - 1:
            optimizer.zero_grad()
            
            train_batch_hazards = torch.cat(train_batch_hazards, dim=0)
            surv_loss = CoxSurvLoss(hazards = train_batch_hazards,
                                    time = train_batch_surv_time,
                                    censorship = train_batch_censorship)

            loss = surv_loss
            train_all_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            torch.cuda.empty_cache()
            train_batch_hazards = []
            train_batch_surv_time = []
            train_batch_censorship = []

    train_all_loss = train_all_loss / (len(patients)//args.batch_size)
    train_ci = calculate_ci(train_patients_surv_time, train_patients_hazards, train_patients_censorship)
        
    return train_all_loss, train_ci
    


def prediction(model,patients,slide_feats_dict,cli_dict,args):
    val_all_hazards = []
    val_all_surv_time = []
    val_all_censorship = []


    val_patients_hazards = {}
    val_patients_surv_time = {}
    val_patients_censorship = {}
    
    model.eval()
    with torch.no_grad():
        for n_iter, patient_id in tqdm(enumerate(patients)):
            surv_time = cli_dict[patient_id][1]
            censorship = cli_dict[patient_id][0]
            val_all_surv_time.append(surv_time)
            val_all_censorship.append(censorship)
            val_patients_surv_time[patient_id] = surv_time
            val_patients_censorship[patient_id] = censorship
            if args.model_name in ['PatchGCN','DeepGraphConv']:
                slide_feats_dict[patient_id].x = slide_feats_dict[patient_id].x[:,:1024]
            else:
                slide_feats_dict[patient_id] = slide_feats_dict[patient_id][:,:1024]
            slide_data = slide_feats_dict[patient_id].to(device)
            
            hazards = model(slide_data)

            hazards = hazards.squeeze(0)
            val_all_hazards.append(hazards)
            val_patients_hazards[patient_id] = hazards.detach().cpu().numpy()
    
    val_all_hazards = torch.cat(val_all_hazards, dim=0)

            
    val_surv_loss = CoxSurvLoss(hazards=val_all_hazards,
                                time=val_all_surv_time,
                                censorship=val_all_censorship)
    
    val_all_loss = val_surv_loss
    val_all_loss = val_all_loss.item()
    val_ci = calculate_ci(val_patients_surv_time, val_patients_hazards, val_patients_censorship)
    
    return val_all_loss, val_ci, val_patients_hazards



def main(args):
    os.makedirs(args.result_dir, exist_ok=True)
    save_dir_name = "{} {} {}_lr_{}_b_{}_fd_{}_epoch_{}_seed_{}_dc_{}_fusion_{}".format(args.cancer_type,args.details,args.model_name,args.lr,args.batch_size,args.fusion_dim,args.epochs,args.seed,args.dc_loss_ratio,args.fusion_loss_ratio)
    save_dir = os.path.join(args.result_dir,save_dir_name)
    os.makedirs(save_dir, exist_ok=True)
    log_name = os.path.join(save_dir, "train_info.log")
    
    f_ = open(log_name,'w')
    f_.truncate()
    
    cli_dict = joblib.load('/path_to/kirc_sur_and_time.pkl')
    train_val_dict = joblib.load('/path_to/kirc_five_folds.pkl')  
    print('load KIRC...')
    if args.model_name in ['PatchGCN','DeepGraphConv']:
        if args.cancer_type == "KIRC_Dino":
            slide_feats_dict = joblib.load('/path_to/kirc_Dino_512_graphs.pkl')
        if args.cancer_type == "KIRC_KimiaNet":
            slide_feats_dict = joblib.load('/path_to/kirc_KimiaNet_512_graphs.pkl')
        if args.cancer_type == "KIRC_MAE":
            slide_feats_dict = joblib.load('/path_to/kirc_MAE_512_graphs.pkl') 
        if args.cancer_type == "KIRC_SimCLR":
            slide_feats_dict = joblib.load('/path_to/kirc_SimCLR_512_graphs.pkl') 
        if args.cancer_type == "KIRC_H-H-MGDM":
            slide_feats_dict = joblib.load('/path_to/kirc_H-H-MGDM_512_graphs.pkl') 
        if args.cancer_type == "KIRC_wo_noisy_t":
            slide_feats_dict = joblib.load('/path_to/kirc_wo_noisy_t_512_graphs.pkl')
        if args.cancer_type == "KIRC_wo_skip":
            slide_feats_dict = joblib.load('/path_to/kirc_wo_skip_512_graphs.pkl')
        if args.cancer_type == "KIRC_DiffAE":
            slide_feats_dict = joblib.load('/path_to/kirc_DiffAE_512_graphs.pkl')
        if args.cancer_type == "KIRC_DiffMAE":
            slide_feats_dict = joblib.load('/path_to/kirc_DiffAE_512_graphs.pkl')
        if args.cancer_type == "KIRC_GraphMAE2":
            slide_feats_dict = joblib.load('/path_to/kirc_GraphMAE2_512_graphs.pkl')
    else:
        if args.cancer_type == "KIRC_Dino":
            slide_feats_dict = joblib.load('/path_to/kirc_Dino_512_features.pkl')
        if args.cancer_type == "KIRC_KimiaNet":
            slide_feats_dict = joblib.load('/path_to/kirc_KimiaNet_512_features.pkl')
        if args.cancer_type == "KIRC_MAE":
            slide_feats_dict = joblib.load('/path_to/kirc_MAE_512_features.pkl')
        if args.cancer_type == "KIRC_SimCLR":
            slide_feats_dict = joblib.load('/path_to/kirc_SimCLR_512_features.pkl')
        if args.cancer_type == "KIRC_H-H-MGDM":
            slide_feats_dict = joblib.load('/path_to/kirc_H-H-MGDM_512_features.pkl')
        if args.cancer_type == "KIRC_wo_noisy_t":
            slide_feats_dict = joblib.load('/path_to/kirc_GraphMAE2_512_features.pkl')
        if args.cancer_type == "KIRC_wo_skip":
            slide_feats_dict = joblib.load('/path_to/kirc_GraphMAE2_512_features.pkl')
        if args.cancer_type == "KIRC_DiffAE":
            slide_feats_dict = joblib.load('/path_to/kirc_DiffAE_512_features.pkl')
        if args.cancer_type == "KIRC_DiffMAE":
            slide_feats_dict = joblib.load('/path_to/kirc_DiffAE_512_features.pkl')
        if args.cancer_type == "KIRC_GraphMAE2":
            slide_feats_dict = joblib.load('/path_to/kirc_GraphMAE2_512_features.pkl')
    print('Load ok')
    all_folds_test_hazards = {}
    for fi in range(args.folds_num):
        setup_seed(args.seed)
        if args.model_name=='AMIL':
            model = AMIL_Surv(in_feats=args.slide_in_feats, size_arg=args.slide_size_arg, dropout = args.dropout, fusion_dim = args.fusion_dim,output_dim = args.output_dim).to(device)
        elif args.model_name=='DeepGraphConv':
            model = DeepGraphConv_Surv(edge_agg='spatial', num_features=args.slide_in_feats, hidden_dim=256, linear_dim=args.slide_size_arg, dropout=args.dropout, n_classes=args.output_dim,device=device).to(device)
        elif args.model_name=='PatchGCN':
            model = PatchGCN_Surv(num_features=args.slide_in_feats, dropout = args.dropout, hidden_dim = args.fusion_dim, n_classes = args.output_dim).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        
        train_patients = train_val_dict["train_{}".format(fi)]
        val_patients = train_val_dict["val_{}".format(fi)]
        test_patients = train_val_dict["test_{}".format(fi)]
        
        print_info(log_name, "---------Fold {}, {} train patients, {} val patients---------\n".format(fi,len(train_patients),len(val_patients)))
        
        best_val_ci = 0.
        best_val_loss = 99.
        for epoch in range(args.epochs):
            if args.if_adjust_lr:
                adjust_learning_rate(optimizer, args.lr, epoch, lr_step=40, lr_gamma=0.5)
            
            train_loss, train_ci = train_one_epoch(model,train_patients,slide_feats_dict,cli_dict,optimizer,args)
            val_loss, val_ci, _ = prediction(model,val_patients,slide_feats_dict,cli_dict,args)
            test_loss, test_ci, _ = prediction(model,test_patients,slide_feats_dict,cli_dict,args)
            
            if epoch>=round(args.epochs*0.2) and epoch <= args.epochs - 1 and val_ci>best_val_ci:
                best_model = copy.deepcopy(model)
                best_val_ci = val_ci
                
            print_info(log_name, "Epoch {:03d}----train loss: {:4f}, train ci: {:4f}, val loss: {:4f}, val ci: {:4f}, test loss: {:4f}, test ci: {:4f}\n".format(epoch,train_loss,train_ci,val_loss,val_ci,test_loss,test_ci))
           
        torch.save(best_model.state_dict(), os.path.join(save_dir,"fold{}_best_model.pth".format(fi)))
        best_model.eval()
        t_test_loss, t_test_ci, t_test_hazards = prediction(best_model,test_patients,slide_feats_dict,cli_dict,args)
        print_info(log_name, "---------Fold {}----test loss: {:4f}, test ci: {:4f}---------\n\n".format(fi,t_test_loss,t_test_ci))
        
        for patient_id in t_test_hazards:
            all_folds_test_hazards[patient_id] = t_test_hazards[patient_id]
        
        del model, best_model, train_patients, val_patients
    
    all_folds_val_ci = calculate_all_ci(cli_dict, all_folds_test_hazards)
    print_info(log_name, "All folds val ci: {:4f}\n".format(all_folds_val_ci))
    
    all_folds_val_info = {}
    all_folds_val_info["hazards"] = all_folds_test_hazards
    joblib.dump(all_folds_test_hazards, os.path.join(save_dir,"test_all_folds_results.pkl"))
    
    f_.close()

if __name__ == "__main__":
    main(args)       