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

train_batch_size=8
train_num_of_worker=10

val_batch_size=int(train_batch_size/8)
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

val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=val_num_of_worker,
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root= data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix),
        ),
    )

test_dataloader=val_dataloader


custom_hooks = [
    dict(
        type='PipelineSwitchHook',
        switch_epoch=2,
        switch_pipeline=[
            dict(
                type='RandomResize',
                scale=(224, 224),
                ),
            dict(
                type='RandomCrop',
                crop_size=(224, 224),
                ),
            dict(
                type='Pad', size=(224, 224),
                pad_val=dict(img=(114, 114, 114))),
        ],
        )
]
val_evaluator = dict(
    type='CocoMetric',
    ann_file=f'{data_root}{val_ann_file}')

test_evaluator=val_evaluator