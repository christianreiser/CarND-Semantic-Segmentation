"""
use this script to change filenames from airsim recording to a format that semantic-segmentation.py can use 
"""

import os
import numpy as np

myimages = [] #list of image filenames

dirFiles = os.listdir('data/data_airsim/training/image_2/.') #list of directory files
dirFiles.sort() #good initial sort but doesnt sort numerically very well
sorted(dirFiles) #sort numerically in ascending order

for files in dirFiles: #filter out all non jpgs
    if '.png' in files:
        myimages.append(files)


def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]

scene, segmentation = split_list(myimages)

if len(scene) != len(segmentation):

    print('len scene != len(seg)')

print('\n',len(scene),'\n\nseg',len(segmentation))

i = 0
for filename in scene:
    dst = str(i) + ".png"
    src = 'data/data_airsim/training/image_2/' + filename
    dst = 'data/data_airsim/training/image_2/' + dst

    # rename() function will
    # rename all the files
    os.rename(src, dst)

    i += 1

i = 0
for filename in segmentation:
    dst = str(i) + ".png"
    src = 'data/data_airsim/training/image_2/' + filename
    dst = 'data/data_airsim/training/gt_image_2/' + dst

    # rename() function will
    # rename all the files
    os.rename(src, dst)

    i += 1

