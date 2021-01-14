import os
import sys

import scipy as sp
import cv2

import imreg_dft as ird

argv = sys.argv
# the TEMPLATE
im0 = cv2.imread(str(argv[1]), cv2.IMREAD_GRAYSCALE).astype('uint8')
# the image to be transformed
im1 = cv2.imread(str(argv[2]), cv2.IMREAD_GRAYSCALE).astype('uint8')

# print("im1-shape{}, dtype{}".format(im1.shape, im1.dtype))

# def similarity(im0, im1, numiter=1, order=3, constraints=None,
#                filter_pcorr=0, exponent='inf', reports=None)

result = ird.similarity(im0, im1, numiter=3, filter_pcorr=3)
#  Returns:
#         dict: Contains following keys: ``scale``, ``angle``, ``tvec`` (Y, X),
#         ``success`` and ``timg`` (the transformed subject image)

report_text = "scale: {0:5g} +-{1:4g}; angle: {2:6g} +-{3:5g}\n"
report_text = report_text + "shift (x, y): {4:6g}, {5:6g}; success: {6:4g}\n"

label = report_text.format(result['scale'], result['Dscale'], result['angle'], result['Dangle']
                           , result['tvec'][0], result['tvec'][1], result['success']
                           )
print(label)

assert "timg" in result
# Maybe we don't want to show plots all the time
if os.environ.get("IMSHOW", "yes") == "yes":
    # import matplotlib.pyplot as plt

    fig = ird.imshow(im0, im1, result['timg'])
    fig.suptitle("{}\n{}".format(label, image_names))
    fig.show()
