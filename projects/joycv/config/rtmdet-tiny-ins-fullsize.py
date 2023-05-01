_base_ = ['mmdet::rtmdet/rtmdet-ins_tiny_8xb32-300e_coco.py']

import platform,os                                                                                                                                                                                                     
                                                                                                                                                                                                                    
if platform.system() == 'Windows':                                                                                                                                                                                  
        data_root = 'B:/opt/hz/'                                                                                                                                                                                    
elif platform.system() == 'Linux':                                                                                                                                                                                  
        data_root = './data/unziped_path/DatasetId_1824199_1682959442/'                                                                                                                                                                           
else:                                                                                                                                                                                                               
    raise NotImplementedError("Unsupported operating system")                                                                                                                                           
       
# Path of train annotation file
train_ann_file = 'Annotations/train.json'
train_data_prefix = 'Images'  # Prefix of train image path
# Path of val annotation file
val_ann_file = 'Annotations/test.json'
val_data_prefix = 'Images'  # Prefix of val image path
def extract_category_info(annotation_path):
    import json
    annotation_info=None
    annotation_info=json.load(open(annotation_path))
    result = []
    categories_info = annotation_info['categories']
    category_info_sorted = sorted(categories_info, key=lambda d: d['id'])
    for cat_item in category_info_sorted:
            result.append(cat_item['name'])
    if len(result) ==0:
        raise Exception(f"Annotation classes is empty. check annotation file {annotation_path}")
    return result

classes=extract_category_info(f"{data_root}{train_ann_file}")
num_classes = len(classes)
print(f"num_classes:{num_classes}")
metainfo = dict(classes=classes, )
print(f"metainfo {metainfo}")
load_from='https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco/rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth'
train_batch_size=4
train_num_of_worker=10

max_epoch=3000
stage2_num_epochs=max_epoch-100

val_batch_size=4
val_num_of_worker=2

stg1_train_size_factor_width=640
stg1_train_size_factor_height=640
stg2_train_size_width=stg1_train_size_factor_width
stg2_train_size_height=stg1_train_size_factor_height
eval_size_width=stg1_train_size_factor_width
eval_size_height=stg1_train_size_factor_height
train_scale_factor=1

model = dict(
    bbox_head=dict(
        type='RTMDetInsSepBNHead',
        num_classes=num_classes)
)


train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, _scope_='mmdet'),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False,
        _scope_='mmdet'),
    dict(
        type='CachedMosaic',
        img_scale=(stg1_train_size_factor_width*train_scale_factor, stg1_train_size_factor_height*train_scale_factor),
        pad_val=114.0,
        max_cached_images=20,
        random_pop=False,
        _scope_='mmdet'),
    dict(
        type='RandomResize',
        scale=(stg1_train_size_factor_width*train_scale_factor, stg1_train_size_factor_height*train_scale_factor),
        ratio_range=(0.5, 2.0),
        keep_ratio=True,
        _scope_='mmdet'),
    dict(type='RandomCrop', crop_size=(stg1_train_size_factor_width, stg1_train_size_factor_height), _scope_='mmdet'),
    dict(type='YOLOXHSVRandomAug', _scope_='mmdet'),
    dict(type='RandomFlip', prob=0.5, _scope_='mmdet'),
    dict(
        type='Pad',
        size=(stg1_train_size_factor_width, stg1_train_size_factor_height),
        pad_val=dict(img=(114, 114, 114)),
        _scope_='mmdet'),
    dict(
        type='CachedMixUp',
        img_scale=(stg1_train_size_factor_width, stg1_train_size_factor_height),
        ratio_range=(1.0, 1.0),
        max_cached_images=10,
        random_pop=False,
        pad_val=(114, 114, 114),
        prob=0.5,
        _scope_='mmdet'),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), _scope_='mmdet'),
    dict(type='PackDetInputs', _scope_='mmdet')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, _scope_='mmdet'),
    dict(type='Resize', scale=(eval_size_width, eval_size_height), keep_ratio=True, _scope_='mmdet'),
    dict(
        type='Pad',
        size=(eval_size_width, eval_size_height),
        pad_val=dict(img=(114, 114, 114)),
        _scope_='mmdet'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'),
        _scope_='mmdet')
]
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=train_num_of_worker,
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        pipeline=train_pipeline,
        data_root= data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        ),
    )
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
        scale=(stg2_train_size_width, stg2_train_size_height),
        ratio_range=(1, 1),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=(stg2_train_size_width, stg2_train_size_height),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(stg2_train_size_width, stg2_train_size_height), pad_val=dict(img=(114, 114, 114))),
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
    metric=['segm'],
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
if(wandb_name==""):
    wandb_name=data_root.split("/")[-2]
wandb_name=f"dataset{wandb_name}-batchsize_{train_batch_size}-maxep_{max_epoch}-stg2_{stage2_num_epochs}-train_scale_factor_{train_scale_factor}-imgscl_{stg2_train_size_width}-evl_{val_evaluator['metric']}-{platform.node()}"
visualizer = dict(
    dict(type='DetLocalVisualizer'),
    vis_backends = [
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'), 
        dict(type='WandbVisBackend',
        init_kwargs={'project': f'rtmdet-tiny-fullsize','name':f"{wandb_name}"},)
    ]) # noqa

