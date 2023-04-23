_base_ = ['mmdet::rtmdet/rtmdet-ins_tiny_8xb32-300e_coco.py']

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

classes=('super','crack',)
num_classes = len(classes)
print(f"num_classes:{num_classes}")
metainfo = dict(classes=classes, )
print(f"metainfo {metainfo}")
load_from='https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco/rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth'
train_batch_size=8
train_num_of_worker=10

max_epoch=3000
stage2_num_epochs=max_epoch-100

val_batch_size=32
val_num_of_worker=2

model = dict(
    bbox_head=dict(
        type='RTMDetInsSepBNHead',
        num_classes=num_classes)
)

train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=train_num_of_worker,
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root= data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        ),
    )


test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, _scope_='mmdet'),
    dict(type='Resize', scale=(224, 224), keep_ratio=True, _scope_='mmdet'),
    dict(
        type='Pad',
        size=(224, 224),
        pad_val=dict(img=(114, 114, 114)),
        _scope_='mmdet'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'),
        _scope_='mmdet')
]

val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=val_num_of_worker,
    dataset=dict(
        type='CocoDataset',
        pipeline=test_pipeline,
        metainfo=metainfo,
        data_root= data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix),
        ),
    )

test_dataloader=val_dataloader

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='RandomResize',
        scale=(224, 224),
        ratio_range=(1, 1),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=(224, 224),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(224, 224), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]
custom_hooks = [
    dict(
        type='PipelineSwitchHook',
        switch_epoch=stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2,
        ),
    dict(type='NumClassCheckHook')    
]
val_evaluator = dict(
    type='CocoMetric',
    ann_file=f'{data_root}{val_ann_file}')

test_evaluator=val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epoch,
    val_interval=10,
    dynamic_intervals=[(stage2_num_epochs, 1)],
    _scope_='mmdet')

default_hooks = dict(
    visualization=dict(type='DetVisualizationHook', draw=True),
    checkpoint=dict(interval=10, max_keep_ckpts=3, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5)
    )
import platform
wandb_name=data_root.split("/")[-1]
visualizer = dict(
    dict(type='DetLocalVisualizer'),
    vis_backends = [
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'), 
        dict(type='WandbVisBackend',
        init_kwargs={'project': f'rtmdet-tiny-{platform.node()}','name':f"{wandb_name}"},)
    ]) # noqa

