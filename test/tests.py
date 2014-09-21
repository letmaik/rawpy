from __future__ import division, print_function

import numpy as np
from numpy.testing.utils import assert_array_equal

import rawpy

def testFileOpen():
    raw = rawpy.imread(r'C:\Users\maik\Desktop\iss\iss030e122539.NEF')
    assert_array_equal(raw.rawdata.shape, [2844, 4288])
    print(np.min(raw.rawdata), np.max(raw.rawdata))
    raw.raw2image()
    # TODO what is the 4th channel??
    for img_channel in raw.image:
        print(img_channel.shape)
        print(np.min(img_channel), np.max(img_channel), np.mean(img_channel))