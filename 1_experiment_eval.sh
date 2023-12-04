pwd
echo '----------start-----------'
echo 'ModelNet40_eval:'
python eval.py --cfg experiments/test_RGM_Seen_Clean_modelnet40_transformer.yaml
python eval.py --cfg experiments/test_RGM_Seen_Jitter_modelnet40_transformer.yaml
python eval.py --cfg experiments/test_RGM_Seen_Crop_modelnet40_transformer.yaml
python eval.py --cfg experiments/test_RGM_Unseen_Crop_modelnet40_transformer.yaml



