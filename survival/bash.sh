python -u train.py --model_name AMIL --cancer_type KIRC_H-MGDM --slide_in_feats 1024 --gpu 5 --batch_size 16
python -u train.py --model_name PatchGCN --cancer_type KIRC_H-MGDM --slide_in_feats 1024 --gpu 5 --batch_size 16
python -u train.py --model_name DeepGraphConv --cancer_type KIRC_H-MGDM --slide_in_feats 1024 --gpu 5 --batch_size 16
