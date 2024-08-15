#Pretrain 
python -u train.py config.yml mask_ratio step;
#Eval
python -u eval.py <ckpt.pth>