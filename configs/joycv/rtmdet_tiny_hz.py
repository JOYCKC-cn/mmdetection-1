_base_ = ['../rtmdet-ins_tiny_8xb32-300e_coco.py']

data_root = '/opt/images/hz/'
# Path of train annotation file
train_ann_file = 'train/_annotations.coco.json'
train_data_prefix = 'train/'  # Prefix of train image path
# Path of val annotation file
val_ann_file = 'valid/_annotations.coco.json'
val_data_prefix = 'valid/'  # Prefix of val image path

classes=('super','crack',)
num_classes = len(classes)
print(f"num_classes:{num_classes}")
metainfo = dict(classes=classes, palette=[(20,40,80),(120,140,80)])
print(f"metainfo {metainfo}")


model = dict(
    bbox_head=dict(
        type='RTMDetInsSepBNHead',
        num_classes=num_classes)
)