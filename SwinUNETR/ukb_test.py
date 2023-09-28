import os
import shutil
import tempfile
import numpy as np

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
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.nets import SwinUNETR
# from monai.data import ITKReader as ITKreader

from monai.handlers import (
    StatsHandler,
    TensorBoardStatsHandler,
    CheckpointSaver,
    CheckpointLoader,
    ValidationHandler,
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
import torch
import nibabel as nib

print_config()


directory = "/home/arda/research-contributions/SwinUNETR/BRATS21"
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SwinUNETR(
    img_size=(96, 96, 96),
    in_channels=1,
    out_channels=46,
    feature_size=48,
    use_checkpoint=True,
).to(device)

model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model_first_train.pth")))

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

data_dir = "data/"
split_json = "dataset.json"
# split_json = "data_100.json"
datasets =  split_json

val_files = load_decathlon_datalist(datasets, True, "test")
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=20, cache_rate=1.0, num_workers=4)
val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)






# with torch.no_grad():
#     print("file name", val_ds[0])
#     img_name = os.path.split(val_ds[0]["image"].meta["filename_or_obj"])[1]
#     img = val_ds[0]["image"]
#     label = val_ds[0]["label"]
#     val_inputs = torch.unsqueeze(img, 1).cuda()
#     val_labels = torch.unsqueeze(label, 1).cuda()
#     val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model, overlap=0.8)
#     # Create a new figure
#     fig = plt.figure("check", (18, 6))
#     plt.subplot(1, 3, 1)
#     plt.title("image")
#     plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, 60], cmap="gray")
#     plt.subplot(1, 3, 2)
#     plt.title("gold label")
#     plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, 60], cmap="Paired")
#     plt.subplot(1, 3, 3)
#     plt.title("prediction")
#     plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, 60], cmap="Paired")

#     # Save the entire figure to a file
#     fig.savefig('output_figure.png')

#     # Close the plot to free up memory
#     plt.close(fig)


## ONLY THE PREDICTION AND THE GOLD LABEL
with torch.no_grad():
    print("file name", val_ds[0])
    img_name = os.path.split(val_ds[0]["image"].meta["filename_or_obj"])[1]
    img = val_ds[0]["image"]
    label = val_ds[0]["label"]
    val_inputs = torch.unsqueeze(img, 1).cuda()
    val_labels = torch.unsqueeze(label, 1).cuda()
    val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model, overlap=0.8)
    # Create a new figure
    fig = plt.figure("check", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("gold label")
    plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, 60], cmap="Paired")
    plt.subplot(1, 2, 2)
    plt.title("prediction")
    plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, 60], cmap="Paired")

    # Save the entire figure to a file
    fig.savefig('output_figure.png')

    # Close the plot to free up memory
    plt.close(fig)




print(label.shape)#unsqueeze 
print(val_labels.shape)

labels = torch.softmax(val_labels, 1).cpu().numpy()
labels = np.argmax(labels, axis =1).astype(np.uint8)[0]

print(val_outputs.shape)
outputs = torch.softmax(val_outputs, 1).cpu().numpy()
outputs = np.argmax(outputs, axis=1).astype(np.uint8)[0]

print(outputs.shape)

#affine = val_ds[case_num]['image_meta_dict']['original_affine']
#print(affine)

inputs = val_inputs.cpu().numpy()[0,0,:,:,:]
print(inputs.shape)

labels = val_labels.cpu().numpy()[0,0,:,:,:]

print(labels.shape)


# path = "/home/tuna/UNETR/BTCV/UKB_inference"
# #outputs=val_outputs.reshape(1,46,171,171,128).astype(np.int8)
# #new_dtype = np.int8  # for example to cast to int8.
# #new_data = new_data.astype(new_dtype)
# label_img = nib.load(os.path.join("/home/tuna/UNETR/BTCV/UKBiobank_dataset/FS_out/FS_out/1884058_20252_2_0/mri", "aparc+aseg.nii"))
# print(label_img.affine)

# nifti_img = nib.Nifti1Image(outputs, label_img.affine)
# nib.save(nifti_img, os.path.join(directory, '6th_test_output.nii'))


# path = "/home/tuna/UNETR/BTCV/UKB_inference"
# #outputs=val_outputs.reshape(1,46,171,171,128).astype(np.int8)
# #new_dtype = np.int8  # for example to cast to int8.
# #new_data = new_data.astype(new_dtype)
# orig_img = nib.load(os.path.join("/home/tuna/UNETR/BTCV/UKBiobank_dataset/FS_out/FS_out/1884058_20252_2_0/mri", "orig.nii"))
# print(orig_img.affine)

# nifti_img = nib.Nifti1Image(inputs, orig_img.affine)
# nib.save(nifti_img, os.path.join(directory, '6th_test_input.nii'))


# path = "/home/tuna/UNETR/BTCV/UKB_inference"
# #outputs=val_outputs.reshape(1,46,171,171,128).astype(np.int8)
# #new_dtype = np.int8  # for example to cast to int8.
# #new_data = new_data.astype(new_dtype)
# gtlabel_img = nib.load(os.path.join("/home/tuna/UNETR/BTCV/UKBiobank_dataset/FS_out/FS_out/1884058_20252_2_0/mri", "orig.nii"))
# print(gtlabel_img.affine)

# nifti_img = nib.Nifti1Image(labels, gtlabel_img.affine)
# nib.save(nifti_img, os.path.join(directory, '6th_test_gtlabels.nii'))