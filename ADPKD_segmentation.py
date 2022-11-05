import argparse
import requests
import shutil
import os
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from nnunet.inference.predict import predict_from_folder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_study_link", "-i", help="http link which is used for downloading study",
                        action="store_true")

    # Dwonload study dicom in format of zip
    download_api = 'http://3.229.206.242/data/projects/pkd/subjects/XNAT_S00009/experiments/XNAT_E00009/scans/ALL/files?format=zip'
    response = requests.get(download_api, auth=('admin', 'admin'))
    with open("study_dicom.zip", "wb") as f:
        f.write(response.content)

    # Extract dicom files for segmentation
    extract_folder = 'tmp/extraction'
    os.makedirs(extract_folder, exist_ok=True)
    shutil.unpack_archive("study_dicom.zip", extract_folder, 'zip')

    # Prepare folder list
    dicom_folder_path_list = [os.path.join(extract_folder,'1000381300/scans/2-3_75_STD/resources/DICOM/files')]

    tmp_nii_folder_path = os.path.join('tmp', 'tmp_nii')
    tmp_output_folder_path = os.path.join('tmp', 'tmp_inference')

    img_output_path_list = [os.path.join(tmp_nii_folder_path, 'ADPKDROUND2_{0:03d}_0000.nii.gz'.format(i)) for i in range(len(dicom_folder_path_list))]
    inference_output_path_list = [os.path.join(tmp_output_folder_path, 'ADPKDROUND2_{0:03d}.nii.gz'.format(i)) for i in range(len(dicom_folder_path_list))]

    api_create_resource_path_list = ["http://3.229.206.242/data/projects/pkd/subjects/XNAT_S00009/experiments/XNAT_E00009/scans/2/resources/SEG_NIFTI_ORGAN_SEGMENTATION?format=NIFTI"]
    api_upload_path_list = ["http://3.229.206.242/data/projects/pkd/subjects/XNAT_S00009/experiments/XNAT_E00009/scans/2/resources/SEG_NIFTI_ORGAN_SEGMENTATION/files/model_segmentation.nii.gz"]

    # loop over all cases in the study
    reader = sitk.ImageSeriesReader()
    for dicom_folder_path, img_output_path in zip(dicom_folder_path_list,img_output_path_list):
        # Prepare folder
        shutil.rmtree(tmp_nii_folder_path, ignore_errors=True)
        shutil.rmtree(tmp_output_folder_path, ignore_errors=True)

        os.makedirs(tmp_nii_folder_path, mode=0o777, exist_ok=True)
        os.makedirs(tmp_output_folder_path, mode=0o777, exist_ok=True)

        #######################################################
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder_path)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        # dicom_shape = image.GetSize()
        dicom_data = sitk.GetArrayFromImage(image)
        dicom_data = dicom_data.transpose((2, 1, 0))

        # Saving data
        D = np.array(image.GetDirection()).reshape(3, 3)
        S = np.array(image.GetSpacing())
        T = np.array(image.GetOrigin())

        affine = np.eye(4)
        affine[:3, :3] = D * S
        affine[:3, -1] = T
        affine[:2] = -affine[:2]

        nii_img = nib.Nifti1Image(dicom_data, affine)
        nib.save(nii_img, img_output_path)

    # Do segmentation
    model_folder_name = os.path.join('trained_models')
    input_folder = tmp_nii_folder_path
    output_folder = tmp_output_folder_path
    folds = 'all'
    save_npz= True
    num_threads_preprocessing = 6
    num_threads_nifti_save = 2
    lowres_segmentations = None
    part_id =0
    num_parts =1
    disable_tta = False
    overwrite_existing = False
    mode = "normal"
    all_in_gpu = None
    step_size = 0.5
    chk = 'model_final_checkpoint'
    disable_mixed_precision = False
    predict_from_folder(model_folder_name, input_folder, output_folder, folds, save_npz, num_threads_preprocessing,
                        num_threads_nifti_save, lowres_segmentations, part_id, num_parts, not disable_tta,
                        overwrite_existing=overwrite_existing, mode=mode, overwrite_all_in_gpu=all_in_gpu,
                        mixed_precision=not disable_mixed_precision,
                        step_size=step_size, checkpoint_name=chk)

    # Upload segmentation
    for inference_output_path,api_create_resource_path_list,api_upload_path in zip(inference_output_path_list,api_create_resource_path_list,  api_upload_path_list):
        with requests.put(api_create_resource_path_list, auth=('admin', 'admin')) as response:
            pass
        with requests.put(api_create_resource_path_list.replace('SEG_NIFTI_ORGAN_SEGMENTATION',"MANUAL_SEG_NIFTI_ORGAN_SEGMENTATION"), auth=('admin', 'admin')) as response:
            pass
        with open(inference_output_path,'rb') as f:
            with  requests.put(api_upload_path, auth=('admin', 'admin'), files={'file': f}) as response:
                pass

if __name__ == "__main__":
    main()