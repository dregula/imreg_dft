import os

import scipy as sp
import imageio

import imreg_dft as ird


basedir = os.path.join('..', 'examples')
# the TEMPLATE
im0 = imageio.imread(os.path.join(basedir, "sample1.png"), as_gray=True)
# the image to be transformed
im1 = imageio.imread(os.path.join(basedir, "sample3.png"), as_gray=True)
result = ird.similarity(im0, im1, numiter=3)

assert "timg" in result
# Maybe we don't want to show plots all the time
if os.environ.get("IMSHOW", "yes") == "yes":
    import matplotlib.pyplot as plt
    ird.imshow(im0, im1, result['timg'])
    plt.show()
