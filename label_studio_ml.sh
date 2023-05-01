label-studio-ml start projects/LabelStudio/backend_template --with \
config_file=/opt/workspace/mmdetection-1/work_dirs/rtmdet-tiny-palmdate/20230429_094136/vis_data/config.py \
checkpoint_file=/opt/workspace/mmdetection-1/work_dirs/rtmdet-tiny-palmdate/best_coco_bbox_mAP_epoch_170.pth \
device=cuda:0 \
--port 8003
