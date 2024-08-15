import os 
import yaml
from easydict import EasyDict
import sys 
config_path = "./config/"+str(sys.argv[1])

with open(config_path, 'r') as f:
    config = EasyDict(yaml.safe_load(f))

if str(sys.argv[3])=='direct':
    config.train.logdir='./logs_direct'
    
    mode_1='direct'
elif str(sys.argv[3])=='step':
    config.train.logdir='./logs_step'
    mode_1='step'
os.makedirs(config.train.logdir, exist_ok=True)    
import setproctitle
setproctitle.setproctitle(config.train.proc_name)
import argparse
import shutil
os.environ['CUDA_VISBLE_DEVICES']=str(config.train.gpu)
import torch
torch.set_num_threads(8)
from torch.nn.utils import clip_grad_norm_
device = torch.device(f"cuda:{str(config.train.gpu)}")
from model.GDM import GDM
from datasets.dataset import PathDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, Dataset
import random 
import time 
from utils import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm 
def train_one_epoch(config,model,epoch,train_loader,optimizer,scheduler,logger,writer):
    model.train()
    sum_loss, sum_n = 0, 0
    sum_node_loss, sum_edge_loss, sum_pos_loss = 0, 0, 0
    sum_kl_loss = 0
    with tqdm(total=len(train_loader),desc='Training') as pbar:
        for batch in train_loader:
            batch = batch.to(device)
            if torch.isnan(batch.x.any()) or torch.isnan(batch.edge_attr.any()):
                continue
            kl_loss, node_loss, edge_loss = model(batch,mode_1=mode_1)
            loss = kl_loss*config.train.kl_alpha + node_loss*config.train.node_alpha+edge_loss*config.train.edge_alpha
            optimizer.zero_grad()
            # with torch.autograd.detect_anomaly():
            loss.backward()
            orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()
                        
            sum_loss += loss.item()
            sum_n += 1
            sum_node_loss += node_loss.item()
            sum_edge_loss += edge_loss.item()
            sum_kl_loss += kl_loss.item()
            pbar.set_postfix({'loss': '%.2f' % (loss.item())})
            pbar.update(1)
            del batch
            # torch.cuda
            torch.cuda.empty_cache()
    avg_loss = sum_loss / sum_n
    avg_node_loss = sum_node_loss / sum_n
    avg_edge_loss = sum_edge_loss / sum_n
    avg_pos_loss = sum_pos_loss / sum_n
    avg_kl_loss = sum_kl_loss / sum_n
    if epoch!=0 and epoch % config.train.val_freq == 0:
        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_node_loss*config.train.node_alpha+avg_edge_loss*config.train.edge_alpha)
        else:
            scheduler.step()
    logger.info(f"[Train] Epoch {epoch:05d} | Loss {avg_loss:.4f} | LR {optimizer.param_groups[0]['lr']:.6f} |")
    logger.info(f"  Loss_node {avg_node_loss:.4f} | Loss_edge {avg_edge_loss:.4f} |")
    logger.info(f"  Loss_KL {avg_kl_loss:.4f} ")

    writer.add_scalar('train/loss', avg_loss, epoch)
    writer.add_scalar('train/loss_node', avg_node_loss, epoch)
    writer.add_scalar('train/loss_edge', avg_edge_loss, epoch)
    writer.add_scalar('train/loss_KL', avg_kl_loss, epoch)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
    writer.flush()
            

config_name = '%s_Graph_Diffusion_MAE' % config.dataset.name
if str(sys.argv[2])=="layer":
    log_dir = get_new_log_dir(config.train.logdir,prefix=config_name)+f"layer_{config.model.all_num_layers}"
elif str(sys.argv[2])=="mask_ratio":
    log_dir = get_new_log_dir(config.train.logdir,prefix=config_name)+f"mask_ratio_{config.model.node_mask_ratio}"

if not os.path.exists(os.path.join(log_dir, 'models')):
    os.makedirs(os.path.join(log_dir, 'models'), exist_ok=True)
ckpt_dir = os.path.join(log_dir, 'checkpoints')
os.makedirs(ckpt_dir, exist_ok=True)
logger = get_logger('train', log_dir)
writer = SummaryWriter(log_dir)
logger.info(config)
logger.info('Loading %s datasets...' % (config.dataset.name))

train_set = PathDataset(root=config.dataset.train)
train_loader = DataLoader(train_set, batch_size=config.train.batch_size, num_workers=config.train.num_workers, pin_memory=True, shuffle=True)
print(len(train_set))
logger.info("Building model...")
model = GDM(config.model).to(device)
init_weights(model)
optimizer = get_optimizer(config.train.optimizer, list(model.parameters()))
scheduler = get_scheduler(config.train.scheduler, optimizer)

for ei in range(0,config.train.max_iters + 1):
    start_time = time.time()
    avg_loss = train_one_epoch(config, model, ei, train_loader, optimizer, scheduler, logger, writer)
    end_time = (time.time() - start_time)
    print('each iteration requires {} s'.format(end_time-start_time))
    if ei!=0 and ei%config.train.save_freq==0:
        ckpt_path = os.path.join(ckpt_dir, '%d.pt' % ei)
        torch.save({
            'config':config,
            'model':model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': ei,
            'avg_loss': avg_loss
        },ckpt_path)
        print('Successfully saved the model!')