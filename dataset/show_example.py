import h5py
import matplotlib.pyplot as plt
import random

i = random.randint(0, 1999)
f = h5py.File('dataset.hdf5', 'r')
celeba = f['celeba'][i]
masked_face = f['masked_faces'][i]
seg_mask = f['segmentation_masks'][i]

f, axarr = plt.subplots(1,3)
axarr[0].imshow(celeba)
axarr[1].imshow(masked_face)
axarr[2].imshow(seg_mask, cmap='gray')

plt.show()