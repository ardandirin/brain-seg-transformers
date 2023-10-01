import os
directory_path = "/home/tuna/UNETR/BTCV/UKBiobank_dataset/FS_out/FS_out/"
import numpy as np
import nibabel as nib

label_map = [4,5,7,8,10,11,12,13,14,15,16,17,18,24,26,28,30,31,43,44,46,47,49,50,51,52,53,54,58,60,62,63,72,77,78,79,80,81,82,85,251,252,253,254,255]
label_dict = {4:1, 5:2, 7:3, 8:4, 10:5, 11:6, 12:7, 13:8, 14:9, 15:10, 16:11, 17:12, 18:13, 24:14, 26:15, 28:16, 30:17, 31:18, 43:19, 44:20, 46:21, 47:22, 49:23, 50:24, 51:25, 52:26, 53:27, 54:28, 58:29, 60:30, 62:31, 63:32, 72:33, 77:34, 78:35, 79:36, 80:37, 81:38, 82:39, 85:40, 251:41, 252:42, 253:43, 254:44, 255:45}

def map_label_keys(data):
    slice, row, col = data.shape
    for i in range(slice):
        for j in range(row):
            for k in range(col):
                label_key=data[i,j,k]
                if label_key in label_map:
                    data[i,j,k] = label_dict[label_key]
                else:
                    data[i,j,k] = 0
    return data
    
def convert_mgz2nii(folder_name, mgz_filename, nii_filename, label_mapping = False):
    mri_img = nib.load(os.path.join(folder_name, mgz_filename))
    data=mri_img.get_fdata()
    data=data.astype(int)
    if label_mapping:
        data=map_label_keys(data)
    nifti_img = nib.Nifti1Image(data, mri_img.affine, mri_img.header)
    nib.save(nifti_img, os.path.join(folder_name, nii_filename))
    
def convertAllDirectories():

    directory_path_enc = os.fsencode(directory_path)
    folder_list = os.listdir(directory_path_enc)
    print("Total number of case folders:",len(folder_list))
    count = 0
    for patient in folder_list:
        patient_folder = os.fsdecode(patient)
        if not patient_folder.startswith("300"):
            continue
        count += 1
        try:
            patient_mri_folder = os.path.join(directory_path, patient_folder, "mri")
            if os.path.isdir(patient_mri_folder):   
                convert_mgz2nii(patient_mri_folder, "orig.mgz", "orig.nii")
                #convert_mgz2nii(patient_mri_folder, "aseg.mgz", "aseg.nii", True)
                convert_mgz2nii(patient_mri_folder, "aparc+aseg.mgz", "aparc+aseg.nii", True)
                print("Folder no.",count,patient_mri_folder)
        except Exception as e:
            print(e)
        
    #/home/tuna/UKBiobank_dataset/FS_out/FS_out/1277686_20252_2_0/mri