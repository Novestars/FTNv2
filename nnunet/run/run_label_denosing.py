#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import argparse
import glob
import os
import nibabel as nib
from sklearn.metrics import jaccard_score
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
def calc(prediction,groudtruth):
    prediction_data = nib.load(prediction).get_fdata()
    groudtruth_data = nib.load(groudtruth).get_fdata()

    jaccard_index = jaccard_score(prediction_data.flatten(), groudtruth_data.flatten(), average=None)
    for i,j in zip(jaccard_index[1:],[1,2,3,4]):
        if i < 0.7:
            s = np.unique(groudtruth_data)
            if j in s:
                print(os.path.basename(prediction))
                print(jaccard_index)
                break
    return i

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input", default='/Users/hexinzi/Dropbox/Mac (3)/Desktop/ct_segmentation/validation_raw',help="input folder for validation results")
    parser.add_argument("-g","--groundtruth", default="/Users/hexinzi/Dropbox/Mac (3)/Desktop/ct_segmentation/training_data/converted_for_nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task667_ADPKDALL/labelsTr",help="folder for groundturth")
    #parser.add_argument("-o","--output", help="output folder for denosed labels")
    parser.add_argument("-t","--threshold", default=0.7,help="threshold for filtering out noisy label")

    args = parser.parse_args()

    model_prediction_path = glob.glob(args.input+ '/*.nii.gz')
    gt_path = [os.path.join(args.groundtruth,os.path.basename( i)) for i in model_prediction_path]

    #class_number = np.unique(nib.load(gt_path[0]).get_fdata())

    with ProcessPoolExecutor(max_workers=16) as executor:
        for out in executor.map(calc, model_prediction_path,gt_path):
            pass


if __name__ == "__main__":
    main()
