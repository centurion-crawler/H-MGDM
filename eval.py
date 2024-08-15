import os 
import yaml

from easydict import EasyDict
import sys 

import argparse
import shutil
from utils import *
Diffmodel,config = load_weights(w_path=str(sys.argv[1])) # sys.argv[1] is the model ckpt path
config.train.max_grad_norm = 100
config.num_classes = 5
config.pool_name = 'GAP'
config.train.gpu = 2
config.train.batch_size = 24

os.environ['CUDA_VISBLE_DEVICES']=str(config.train.gpu)
import setproctitle
setproctitle.setproctitle(config.train.proc_name)
import torch
import torch.nn.functional as F
# torch.set_num_threads(8)
from torch.nn.utils import clip_grad_norm_
device = torch.device(f"cuda:{str(config.train.gpu)}")
from model.GDM import GDM
from model.classifier import classifier
from datasets.dataset import PathDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, Dataset
import random 
import time 

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report
from tqdm import tqdm 

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report,accuracy_score,roc_curve,auc,roc_auc_score
from collections import Counter

def return_auc(target_array,possibility_array,num_classes):
    enc = OneHotEncoder()
    target_onehot = enc.fit_transform(target_array.long().unsqueeze(1))
    target_onehot = target_onehot.toarray()
    class_auc_list = []
    for i in range(num_classes):
        # print(target_onehot[:,i].shape,possibility_array[:,i].shape)
        class_i_auc = roc_auc_score(target_onehot[:,i], possibility_array[:,i])
        class_auc_list.append(class_i_auc)
    macro_auc = roc_auc_score(np.round(target_onehot,0), possibility_array, average="macro", multi_class="ovo")
    return macro_auc, class_auc_list

def train_one_epoch(config,Diffmodel,model,epoch,train_loader,optimizer,scheduler,logger,writer):
    Diffmodel.train()
    model.train()
    sum_loss, sum_n = 0, 0
    sum_node_loss, sum_edge_loss, sum_pos_loss = 0, 0, 0
    sum_kl_loss = 0
    weights = [1,1.02,2.2,1.8,5.5]
    cls_weights = torch.FloatTensor(weights).to(device)
    loss_func = nn.CrossEntropyLoss(weight = cls_weights).to(device)
    with tqdm(total=len(train_loader),desc='Training') as pbar:
        for batch, cls_label in train_loader:
            batch = batch.to(device)
            cls_label = cls_label.to(device)
            if torch.isnan(batch.x.any()) or torch.isnan(batch.edge_attr.any()):
                continue
            # with torch.no_grad():
            enc_data, batch_node, batch_edge = Diffmodel.forward_eval(batch,mode_1='step')
            logits = model(enc_data, batch_node, batch_edge)
            loss = loss_func(logits, cls_label)
            loss.backward()
            orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()

            sum_loss += loss.item()
            sum_n += 1
            pbar.set_postfix({'loss': '%.2f' % (loss.item())})
            pbar.update(1)
            del batch
            torch.cuda.empty_cache()

    avg_loss = sum_loss / sum_n
    if epoch!=0 and epoch % config.train.val_freq == 0:
        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        else:
            scheduler.step()
    logger.info(f"[Train] Epoch {epoch:05d} | Loss {avg_loss:.4f} | LR {optimizer.param_groups[0]['lr']:.6f} |")
    writer.add_scalar('train/loss', avg_loss, epoch)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
    writer.flush()

    return avg_loss

def eval_model(config,Diffmodel,model,epoch,eval_loader,optimizer,scheduler,logger,writer,mode='Validing'):
    Diffmodel.eval()
    model.eval()
    sum_loss, sum_n = 0, 0
    sum_node_loss, sum_edge_loss, sum_pos_loss = 0, 0, 0
    sum_kl_loss = 0
    Y_hat_all = []
    Y_all = []
    Y_scores = []
    weights = [1,1.02,2.2,1.8,5.5]
    cls_weights = torch.FloatTensor(weights).to(device)
    loss_func = nn.CrossEntropyLoss(weight = cls_weights).to(device)
    with tqdm(total=len(eval_loader),desc=mode) as pbar:
        for batch, cls_label in eval_loader:
            batch = batch.to(device)
            cls_label = cls_label.to(device)
            if torch.isnan(batch.x.any()) or torch.isnan(batch.edge_attr.any()):
                continue
            # kl_loss, node_loss, edge_loss, pos_loss = Diffmodel.forward_eval(batch)
            with torch.no_grad():
                enc_data, batch_node, batch_edge = Diffmodel.forward_eval(batch,mode_1='step')
                logits = model(enc_data, batch_node, batch_edge)
            loss = loss_func(logits, cls_label)
            Y_hat = list(torch.argmax(logits,dim=-1).detach().cpu().numpy())
            Y_hat_all = Y_hat_all + Y_hat
            Y_scores += list(F.softmax(logits,dim=-1).detach().cpu().numpy())
            Y_all = Y_all + list(cls_label.cpu().numpy())
            sum_loss += loss.item()
            sum_n += 1
            pbar.set_postfix({'loss': '%.2f' % (loss.item())})
            pbar.update(1)
            del batch


    avg_loss = sum_loss / sum_n
    logger.info(f"[{mode}] Epoch {epoch:05d} | Loss {avg_loss:.4f} | LR {optimizer.param_groups[0]['lr']:.6f} |")
    auc = return_auc(torch.Tensor(Y_all), torch.Tensor(Y_scores), num_classes=config.num_classes)
    logger.info(f'marco auc: {auc}')
    logger.info(classification_report(Y_all, Y_hat_all))
    
    if mode == 'Validing':
        writer.add_scalar('valid/loss', avg_loss, epoch)
        writer.add_scalar('valid/lr', optimizer.param_groups[0]['lr'], epoch)
        writer.flush()
    elif  mode == 'Testing':
        writer.add_scalar('test/loss', avg_loss, epoch)
        writer.add_scalar('test/lr', optimizer.param_groups[0]['lr'], epoch)
        writer.flush()
    return avg_loss

config_name = '%s_Graph_Diffusion_MAE' % config.dataset.name
if 'mask_ratio' in str(sys.argv[2]):
    log_dir = get_new_log_dir(config.train.logdir,prefix=config_name)+f"finetune_step_{str(sys.argv[2])}_{config.model.node_mask_ratio}"
elif 'layer' in str(sys.argv[2]):
    log_dir = get_new_log_dir(config.train.logdir,prefix=config_name)+f"finetune_step_{str(sys.argv[2])}_{config.model.all_num_layers}"
if not os.path.exists(os.path.join(log_dir, 'models')):
    os.makedirs(os.path.join(log_dir, 'models'), exist_ok=True)
ckpt_dir = os.path.join(log_dir, 'checkpoints')
os.makedirs(ckpt_dir, exist_ok=True)
logger = get_logger('train', log_dir)
writer = SummaryWriter(log_dir)
logger.info(config)
logger.info('Loading %s datasets...' % (config.dataset.name))


config.dataset.train = "/data2/zzf/Graph_data/TISSUE_GRAPH"
for fi in range(5):
    config.train.batch_size = 128
    config.train.optimizer.lr = 3.e-4
    config.train.num_workers = 4
    train_set = PathDataset(root=config.dataset.train, is_eval=True, mod='train', fold=fi)
    train_loader = DataLoader(train_set, batch_size=config.train.batch_size, num_workers=config.train.num_workers, pin_memory=True, shuffle=True)
    val_set = PathDataset(root=config.dataset.train, is_eval=True, mod='val', fold=fi)
    val_loader = DataLoader(val_set, batch_size=config.train.batch_size, num_workers=config.train.num_workers, pin_memory=True, shuffle=True)
    test_set = PathDataset(root=config.dataset.train, is_eval=True, mod='test', fold=fi)
    test_loader = DataLoader(test_set, batch_size=config.train.batch_size, num_workers=config.train.num_workers, pin_memory=True, shuffle=True)


    # print(len(train_set))
    logger.info("Building model...")

    clsmodel = classifier(config).to(device)
    Diffmodel = Diffmodel.to(device)
    init_weights(clsmodel)
    # optimizer = get_optimizer(config.train.optimizer, [clsmodel.parameters()])
    optimizer = get_optimizer(config.train.optimizer, [clsmodel.parameters(),list(Diffmodel.x_embedding.parameters())+list(Diffmodel.edge_attr_embedding.parameters())+list(Diffmodel.context_encoder.parameters())])
    scheduler = get_scheduler(config.train.scheduler, optimizer)

    Diffmodel = Diffmodel.to(device)
    config.train.max_iters=100
    config.train.save_freq=20

    for ei in range(0,config.train.max_iters + 1):
        start_time = time.time()
        avg_loss = train_one_epoch(config, Diffmodel, clsmodel, ei, train_loader, optimizer, scheduler, logger, writer)
        eval_model(config, Diffmodel, clsmodel, ei, val_loader, optimizer, scheduler, logger, writer, mode='Validing')
        eval_model(config, Diffmodel, clsmodel, ei, test_loader, optimizer, scheduler, logger, writer, mode='Testing')
        end_time = (time.time() - start_time)
        print('each iteration requires {} s'.format(end_time-start_time))
        if ei!=0 and ei%config.train.save_freq==0:
            ckpt_path = os.path.join(ckpt_dir, '%d.pt' % ei)
            torch.save({
                'config':config,
                'clsmodel':clsmodel.state_dict(),
                'Diffmodel':Diffmodel.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': ei,
                'avg_loss': avg_loss
            },ckpt_path)
            print('Successfully saved the model!')
