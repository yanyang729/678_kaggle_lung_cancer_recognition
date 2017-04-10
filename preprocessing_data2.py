# -*- coding: utf-8 -*-
# @Author: Nico Zheng
# @Date:   2017-04-07 19:22:57
# @Last Modified by:   yang
# @Last Modified time: 2017-04-08 21:31:18
#!/usr/local/bin/python


from __future__ import division
import numpy as np
import time
import dicom 
from multiprocessing import Pool, Lock
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from skimage.transform import resize
from glob2 import glob
import pandas as pd
from tqdm import tqdm



# set glob parameters
ct_file_folder = "/media/yang/My Files/bowl_data/stage2/"
saved_folder = "/home/yang/YyProj/bowl/input_submit"
processer = 4


# functions
def load_patient(patient_id, sample_file):
    """
    load patient scans and sort according to SliceLocation
    return a stack numpy array
    """
    files = [c for c in sample_file if patient_id in c]
    tmp = [dicom.read_file(c) for c in files]
    imgs = {}
    for i in tmp:
        tmp2 = i.pixel_array
        tmp2[tmp2 == -2000] = 0
        try:
            imgs[float(i.ImagePositionPatient[2])] = tmp2
        except AttributeError:
            pass
    # sort by location
    imgs_sort = [x[1] for x in sorted(imgs.items(), key=lambda x: x[0])]
    return np.stack(imgs_sort)


def preprocess_ct(one_ct_array):
    """
    preprocess cts
    only conatain gray value less than 604(remove bones and other hard issure)
    only remain largest 2 data areas
    data areas removed will be replaced with 0
    """
    label_image = label(clear_border(one_ct_array < 604))   # select data less than 604 grey value then label it
    areas = [r.area for r in regionprops(label_image)]   # find ct areas
    areas.sort()
    if len(areas) > 2:   # only contain most large 2 areas
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0  # build binary mask
    after = np.where(binary, one_ct_array, 0)   # remove accroding to mask
    return after


def preprocess_patient(patient_id, sample_file):
    """
    pack preprocessing steps
    prepare for multiprocess
    """
    patient_cts = load_patient(patient_id, sample_file)
    preprocessed = np.stack([preprocess_ct(c) for c in patient_cts])
    resized = resize(preprocessed, (20, 50, 50))
    # return resized
    scan_count, col, row = resized.shape
    np.save(saved_folder + "/%s_%s_%s_%s.npy" % (patient_id, scan_count, col, row), resized)
    # print("finished patient %s" % (patient_id))


# def run(patient_ids,sample_file):
#     # n = patient_ids.pop()
#     while patient_ids:
#         n = patient_ids.pop()
#         preprocess_patient(n, sample_file)

# def worker(patient_ids, sample_file, lock):
#     """
#     Prints out the item that was passed in
#     """
#     for item in patient_ids:
#         lock.acquire()
#         try:
#             preprocess_patient(item,sample_file)
#         finally:
#             lock.release()
    

if __name__ == '__main__':
    sample_file = glob(ct_file_folder + '/*/*')
    patient_ids = glob(ct_file_folder + '/*')
    patient_ids = [c.split("/")[-1] for c in patient_ids]
    files_saved_folder = [c.split("/")[-1].split(".")[0].split("_")[0] for c in glob(saved_folder+"/*")]
    patient_ids = list(set(patient_ids) - set(files_saved_folder))
    print ("there are %s of patients need to process" % len(patient_ids))
    t0 = time.time()
    c = 1
    e = 0
    for item in patient_ids:
        try:
            print("patient NO. %s" % c)
            preprocess_patient(item,sample_file)
            c += 1


        except ValueError:
            e += 1
            print ("patient %s is error!" %item)

    t1 = time.time()
    print('we have {} error'.format(e))
    print('all process done with in time %s' %str((t1-t0)/60))
