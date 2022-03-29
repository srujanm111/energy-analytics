import os
from mask_the_face.mask_the_face import mask_the_face
import cv2 as cv
import numpy as np
from random import randint
import h5py

# paths
dataset_dir = 'dataset'
celeba_dir = f'{dataset_dir}/celeba'
processed_files = f'{dataset_dir}/processed_files.txt'
hdf5_path = f'{dataset_dir}/dataset.hdf5'

# clear previously processed files for new run
pftxt = open(processed_files, 'w')

# face mask configurations
mask_type_list=['surgical', 'KN95', 'cloth']
mask_colors_list=['#0473e2', '#000000', '#ffffff', '#1fc49b']

# master lists
celeba_arr = []
masked_faces_arr = []
segmentation_masks_arr = []

for img_name in sorted(os.listdir(celeba_dir)):
    # make dataset size 50000 image trios
    if len(celeba_arr) == 50000: break

    # file processing status
    print(f'\r{img_name}', end='')

    # read and resize the maskless face image
    original_img = cv.imread(f'{celeba_dir}/{img_name}')
    original_img = original_img[5:213, 1:177]
    original_img = cv.resize(original_img, (88, 104), interpolation=cv.INTER_AREA)

    # get the masked face image
    has_face, masked_img = mask_the_face(original_img, mask_type_list[randint(0, 2)], mask_colors_list[randint(0, 3)])

    # if mask_the_face did not overlay a mask, skip this image trio
    if not has_face: continue

    # create the segmentation mask from the pixel difference between maskless and masked images
    seg_mask = original_img.copy()
    cv.absdiff(original_img, masked_img, seg_mask)  # calculate differences for each channel RGB
    seg_mask = np.sum(seg_mask, axis=2)  # sum the channel differences
    seg_mask[seg_mask > 0] = 1  # label a position as masked if a nonzero pixel dif exists

    # format the iamge arrays
    original_img = cv.cvtColor(original_img, cv.COLOR_BGR2RGB)
    masked_img = cv.cvtColor(masked_img, cv.COLOR_BGR2RGB)
    seg_mask = seg_mask.astype('uint8')

    # append to master lists
    celeba_arr.append(original_img)
    masked_faces_arr.append(masked_img)
    segmentation_masks_arr.append(seg_mask)

    # write the image name to processed_files.txt
    pftxt.write(img_name + '\n')

print()
pftxt.close()


# write master lists to HDF5 dataset

celeba_arr = np.array(celeba_arr)
masked_faces_arr = np.array(masked_faces_arr)
segmentation_masks_arr = np.array(segmentation_masks_arr)

print(f'celeba: {celeba_arr.shape}')
print(f'masked_faces: {masked_faces_arr.shape}')
print(f'segmentation_masks: {segmentation_masks_arr.shape}')

f = h5py.File(hdf5_path, 'w')
f.create_dataset('celeba', data=celeba_arr, chunks=(1000, 104, 88, 3))
f.create_dataset('masked_faces', data=masked_faces_arr, chunks=(1000, 104, 88, 3))
f.create_dataset('segmentation_masks', data=segmentation_masks_arr, chunks=(1000, 104, 88))
