pwd
echo '----start train----'
python train.py --cfg experiments/train_RGM_Seen_Clean_modelnet40_transformer.yaml
python train.py --cfg experiments/train_RGM_Seen_Jitter_modelnet40_transformer.yaml
python train.py --cfg experiments/train_RGM_Seen_Crop_modelnet40_transformer.yaml
python train.py --cfg experiments/train_RGM_Unseen_Crop_modelnet40_transformer.yaml

