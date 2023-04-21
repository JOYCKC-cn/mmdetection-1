_base_ = [
    'mmdet::_base_/datasets/coco_instance.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]



data_root = '/opt/images/hz/'
# Path of train annotation file
train_ann_file = 'train/_annotations.coco.json'
train_data_prefix = 'train/'  # Prefix of train image path
# Path of val annotation file
val_ann_file = 'valid/_annotations.coco.json'
val_data_prefix = 'valid/'  # Prefix of val image path

classes=('crack',)

train_batch_size_per_gpu = 24
train_num_workers = 4
val_batch_size_per_gpu = 1
val_num_workers = 2

num_classes = len(classes)+1
print(f"num_classes:{num_classes}")
metainfo = dict(classes=classes, palette=[(20,40,80),(120,140,80)])


training_param_factor=1
input_image_size = (244, 244)
learning_rate = 0.0001
base_batch_size = train_batch_size_per_gpu

# For the backbone
in_channels = [512, 1024, 2048]
out_channels = 256



# For the encoder and decoder
ins_dim = 256/training_param_factor
mask_dim = 256/training_param_factor
kernel_dim = 128/training_param_factor
ins_conv = 4
mask_conv = 4
num_masks = 100
scale_factor=2.0/training_param_factor
#############################
custom_imports = dict(
    imports=['projects.SparseInst.sparseinst'], allow_failed_imports=False)

model = dict(
    type='SparseInst',
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
        out_indices=(1, 2, 3),
        frozen_stages=0,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    encoder=dict(
        type='InstanceContextEncoder',
        in_channels=in_channels,
        out_channels=out_channels
        ),
    decoder=dict(
        type='BaseIAMDecoder',
        num_classes=num_classes,
        in_channels=out_channels + 2,
        ins_dim=ins_dim,
        ins_conv=ins_conv,
        mask_dim=mask_dim,
        mask_conv=mask_conv,
        kernel_dim=kernel_dim,
        num_masks=num_masks,
        scale_factor=scale_factor,
        output_iam=False),
    criterion=dict(
        type='SparseInstCriterion',
        num_classes=num_classes,
        assigner=dict(type='SparseInstMatcher', alpha=0.8, beta=0.2),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            alpha=0.25,
            gamma=2.0,
            reduction='sum',
            loss_weight=2.0),
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=1.0),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            reduction='sum',
            eps=5e-5,
            loss_weight=2.0),
    ),
    test_cfg=dict(score_thr=0.005, mask_thr_binary=0.45)
    )

backend = 'pillow'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args={{_base_.backend_args}},
        imdecode_backend=backend),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='RandomChoiceResize',
        scales=[input_image_size,(640, 853)],
        keep_ratio=True,
        backend=backend),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args={{_base_.backend_args}},
        imdecode_backend=backend),
    dict(type='Resize', scale=input_image_size, keep_ratio=True, backend=backend),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]



train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    sampler=dict(type='InfiniteSampler'),
    dataset=dict(
        pipeline=train_pipeline,
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
        pipeline=test_pipeline,
        data_prefix=dict(img=val_data_prefix)))

test_dataloader = val_dataloader
val_evaluator = dict(metric='segm')
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(_delete_=True, type='AdamW', lr=learning_rate, weight_decay=0.05))
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=270000,
    val_interval=10000)
# learning rate
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=270000,
        by_epoch=False,
        milestones=[210000, 250000],
        gamma=0.1)
]

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=10000, max_keep_ckpts=3))
log_processor = dict(by_epoch=False)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64, enable=True)



