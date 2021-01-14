import os

import scipy as sp
import cv2

import imreg_dft as ird


basedir = os.path.join('..', 'examples')
# the TEMPLATE
im0 = cv2.imread(os.path.join(basedir, "sample1.png"), cv2.IMREAD_GRAYSCALE)
# the image to be transformed
im1 = cv2.imread(os.path.join(basedir, "sample3.png"), cv2.IMREAD_GRAYSCALE)

result = ird.similarity(im0, im1, numiter=3)
assert "timg" in result
# Maybe we don't want to show plots all the time
if os.environ.get("IMSHOW", "yes") == "yes":
    import matplotlib.pyplot as plt
    ird.imshow(im0, im1, result['timg'])
    plt.show()
