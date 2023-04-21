_base_ = [
    'mmdet::_base_/datasets/coco_instance.py',
    'mmdet::_base_/schedules/schedule_1x.py', 'mmdet::_base_/default_runtime.py'
]

import platform                                                                                                                                                                                                     
                                                                                                                                                                                                                    
if platform.system() == 'Windows':                                                                                                                                                                                  
        data_root = 'B:/opt/hz/'                                                                                                                                                                                    
elif platform.system() == 'Linux':                                                                                                                                                                                  
            data_root = '/opt/images/hz/'                                                                                                                                                                           
else:                                                                                                                                                                                                               
                raise NotImplementedError("Unsupported operating system")                                                                                                                                           
       
# Path of train annotation file
train_ann_file = 'train/_annotations.coco.json'
train_data_prefix = 'train/'  # Prefix of train image path
# Path of val annotation file
val_ann_file = 'valid/_annotations.coco.json'
val_data_prefix = 'valid/'  # Prefix of val image path

classes=('crack','crack')

train_batch_size_per_gpu = 8
train_num_workers = 4
val_batch_size_per_gpu = 1
val_num_workers = 2

num_classes = len(classes)
print(f"num_classes:{num_classes}")
metainfo = dict(classes=classes, palette=[(20,40,80),(120,140,80)])
# model settings
model = dict(
    type='SOLOv2',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=5),
    mask_head=dict(
        type='SOLOV2Head',
        num_classes=80,
        in_channels=256,
        feat_channels=512,
        stacked_convs=4,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
        pos_scale=0.2,
        num_grids=[40, 36, 24, 16, 12],
        cls_down_index=0,
        mask_feature_head=dict(
            feat_channels=128,
            start_level=0,
            end_level=3,
            out_channels=256,
            mask_stride=4,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
        loss_mask=dict(type='DiceLoss', use_sigmoid=True, loss_weight=3.0),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0)),
    # model training and testing settings
    test_cfg=dict(
        nms_pre=500,
        score_thr=0.1,
        mask_thr=0.5,
        filter_thr=0.05,
        kernel='gaussian',  # gaussian/linear
        sigma=2.0,
        max_per_img=100))

# optimizer
optim_wrapper = dict(
    optimizer=dict(lr=0.01), clip_grad=dict(max_norm=35, norm_type=2))


train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    sampler=dict(type='InfiniteSampler'),
    dataset=dict(
       
        data_root=data_root,
        metainfo=metainfo,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix)))

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file=val_ann_file,
       
        data_prefix=dict(img=val_data_prefix)))

val_evaluator = dict(
    ann_file=f'{data_root}{val_ann_file}',
    metric='segm',
    format_only=False,
    backend_args=None,
    _scope_='mmdet')
test_dataloader = val_dataloader
test_evaluator = val_evaluator

visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')]) # noqa