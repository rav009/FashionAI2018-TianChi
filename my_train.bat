set num_gpus=1
set num_workers=1
set model=resnet50_v2
set epoch=15
set batch_size=10

set MXNET_CUDNN_AUTOTUNE_DEFAULT=0

set task_file_1=my_train_task_design_windows.py
set task_file_2=my_train_task_length_windows.py

::python3 %task_file_1% --task collar_design_labels --model %model% --batch-size %batch_size% --num-gpus %num_gpus% -j %num_workers% --epochs %epoch% --lr-steps 7,12,15
::python3 %task_file_1% --task neckline_design_labels --model %model% --batch-size %batch_size% --num-gpus %num_gpus% -j %num_workers% --epochs %epoch% --lr-steps 7,12,15
::python3 %task_file_1% --task neck_design_labels --model %model% --batch-size %batch_size% --num-gpus %num_gpus% -j %num_workers% --epochs %epoch% --lr-steps 7,12,15
::python3 %task_file_1% --task lapel_design_labels --model %model% --batch-size %batch_size% --num-gpus %num_gpus% -j %num_workers% --epochs %epoch% --lr-steps 7,12,15

python3 %task_file_2% --task skirt_length_labels --model %model% --batch-size %batch_size% --num-gpus %num_gpus% -j %num_workers% --epochs %epoch% --lr-steps 7,12,15

pause



