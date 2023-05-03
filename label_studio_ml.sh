label-studio-ml start projects/LabelStudio/backend_template --with \
config_file=/opt/workspace/mmdetection-1/work_dirs/rtmdet-tiny-ins-fullsize/rtmdet-tiny-ins-fullsize.py \
checkpoint_file=/opt/workspace/mmdetection-1/work_dirs/rtmdet-tiny-ins-fullsize/best_coco_segm_mAP_epoch_130.pth \
device=cuda:0 \
--port 8003
