log_file=logs/run_val.txt
python imagenet.py     -a mobilenetv2     -d /home/dl/DATA/ImageNet/ILSVRC/Data/CLS-LOC     --weight ./pretrained/mobilenetv2_0.5-eaa6f9ad.pth     --width-mult 0.5     --input-size 224  -e 2>&1 | tee -a $log_file