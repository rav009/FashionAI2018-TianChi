num_gpus=0
num_workers=1
model=resnet50_v2 #resnet152_v2
epoch=15
batch_size=16


python3 my_train_task_design_linux.py --task collar_design_labels --model $model --batch-size $batch_size --num-gpus $num_gpus -j $num_workers --epochs $epoch --lr-steps 7,12,15
