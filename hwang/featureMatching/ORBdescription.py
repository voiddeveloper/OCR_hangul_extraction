import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

imgStandard = cv.imread('hwang/imgSet/standard/7_5.png')
imgComparison = cv.imread('hwang/imgSet/test_comparison8.png')

#############################################################################
# orb detector 생성
orb = cv.ORB_create()

# orb를 이용한 이미지 특징점 찾기
kpStandard, desStandard = orb.detectAndCompute(imgStandard, None)
kpComparison, desComparison = orb.detectAndCompute(imgComparison, None)

imgStandard = cv.drawKeypoints(imgStandard, kpStandard, None)
imgComparison = cv.drawKeypoints(imgComparison, kpComparison, None)
cv.imshow('frame1', imgStandard)
cv.imshow('frame2', imgComparison)

cv.waitKey(0)
##############################################################################

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
matches = bf.match(desStandard, desComparison)
matches = sorted(matches, key = lambda x:x.distance)

imgResult = cv.drawMatches(imgStandard, kpStandard, imgComparison, kpComparison, matches[:10], None, flags = 2)
plt.imshow(imgResult), plt.show()