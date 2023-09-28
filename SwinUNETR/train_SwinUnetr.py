import os
import shutil
import tempfile

import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    LoadImage,
    ToTensor,
    Activationsd,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
    AsDiscreted,
    Invertd,
    SaveImaged
)

from monai.config import print_config
from monai.metrics import DiceMetric, meandice
from monai.networks.nets import SwinUNETR
from monai.data import ITKReader as ITKreader

from monai.handlers import (
    StatsHandler,
    TensorBoardStatsHandler,
    CheckpointSaver,
    CheckpointLoader,
    ValidationHandler,
    HausdorffDistance,
    MeanDice
)

from monai.data import (
    Dataset,
    DataLoader,
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)

import numpy as np
import torch

print_config()


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

directory = "/home/arda/research-contributions/SwinUNETR/BRATS21"
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)


num_samples = 4

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# CUDA_VISIBLE_DEVICES=

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        # EnsureChannelFirstd(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=num_samples,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
    ]
)


data_dir = "/home/tuna/UNETR/BTCV/UKBiobank_dataset/FS_out/FS_out/"
split_json = "UKBiobank_data.json"
datasets =  data_dir + split_json

datalist = load_decathlon_datalist(datasets, True, "training")
val_files = load_decathlon_datalist(datasets, True, "validation")
train_ds = CacheDataset(
    data=datalist,
    transform=train_transforms,
    cache_num=24,
    cache_rate=1.0,
    num_workers=2,
)
train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=1, shuffle=True)
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=20, cache_rate=1.0, num_workers=2)
val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1, shuffle=False)

set_track_meta(True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # Use to turn on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = SwinUNETR(
    img_size=(96, 96, 96),
    in_channels=1,
    out_channels=45,
    feature_size=48,
    use_checkpoint=True,
).to(device)


# torch.backends.cudnn.benchmark = True
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scaler = torch.cuda.amp.GradScaler()


# ### Execute a typical PyTorch training process


def validation(epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))
        dice_val_each_label = dice_metric.get_dice_foreach_label()
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val, dice_val_each_label


def train(global_step, train_loader, dice_val_best, global_step_best, dice_val_each_label_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        with torch.cuda.amp.autocast():
            logit_map = model(x)
            loss = loss_function(logit_map, y)
        scaler.scale(loss).backward()
        epoch_loss += loss.item()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        epoch_iterator.set_description(f"Training ({global_step} / {max_iterations} Steps) (loss={loss:2.5f})")
        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            dice_val, dice_val_each_label = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                dice_val_each_label_best = dice_val_each_label
                print("Dice val each label", dice_val_each_label_best)
                global_step_best = global_step
                torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model_with_labels_45.pth"))
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        global_step += 1
    return global_step, dice_val_best, global_step_best, dice_val_each_label_best


max_iterations = 25000
eval_num = 500
post_label = AsDiscrete(to_onehot=46)
post_pred = AsDiscrete(argmax=True, to_onehot=46)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
dice_val_each_label_best = torch.empty(46)
global_step_best = 0
epoch_loss_values = []
metric_values = []
while global_step < max_iterations:
    global_step, dice_val_best, global_step_best, dice_val_each_label_best = train(global_step, train_loader, dice_val_best, global_step_best, dice_val_each_label_best)


print(f"train completed, best_metric: {dice_val_best:.4f} " f"at iteration: {global_step_best}")



with open('epoch_loss_values_with_label_45.txt','w') as fp:
    for item in epoch_loss_values:
        fp.write("%f\n"%item)

with open('metric_values_with_label_45.txt','w') as fp:
    for item in metric_values:
        fp.write("%f\n"%item)

# Convert tensor to numpy array
numpy_array = dice_val_each_label_best.numpy()

# Write to txt file
with open("label_values_45.txt", "w") as f:
    for value in numpy_array:
        f.write(str(value) + '\n')
