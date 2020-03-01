import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from numpy.fft import fft, ifft
import cv2
import random


img = cv2.imread('m_B148-2.png')
img=cv2.resize(img,(1700,700))
# cv2.imshow('img',img)
# cv2.waitKey()

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()