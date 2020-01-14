import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

imgStandard = cv.imread('hwang/imgSet/test1.png')
# imgComparison = cv.imread('hwang/imgSet/ant+hill_7_3.jpg', 0)
# imgComparisonRGB = cv.imread('hwang/imgSet/ant+hill_7_3.jpg', 1)




# # orb detector 생성
# orb = cv.ORB_create()

# # orb를 이용한 이미지 특징점 찾기
# kpStandard, desStandard = orb.detectAndCompute(imgStandard, None)
# kpComparison, desComparison = orb.detectAndCompute(imgComparison, None)

# #############################################################################
# imgStandard = cv.drawKeypoints(imgStandard, kpStandard, None)
# imgComparison = cv.drawKeypoints(imgComparison, kpComparison, None)
# cv.imshow('frame1', imgStandard)
# cv.imshow('frame2', imgComparison)

# cv.waitKey(0)

# bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
# matches = bf.match(desStandard, desComparison)
# matches = sorted(matches, key = lambda x:x.distance)

# imgResult = cv.drawMatches(imgStandard, kpStandard, imgComparison, kpComparison, matches[:10], None, flags = 2)
# plt.imshow(imgResult), plt.show()
##############################################################################
# bf = cv.BFMatcher()
# matches = bf.knnMatch(desStandard, desComparison, k=2)

# good = []
# for m,n in matches:
#     if m.distance < 0.75 * n.distance:
#         good.append([m])

# imgResult = cv.drawMatchesKnn(imgStandard, kpStandard, imgComparison, kpComparison, good, None, flags=2)
# print(good)
# plt.imshow(imgResult), plt.show()
##############################################################################
# sift = cv.xfeatures2d.SIFT_create()

# kpStandard, desStandard = sift.detectAndCompute(imgStandard, None)
# kpComparison, desComparison = sift.detectAndCompute(imgComparison, None)

# FLANN_INDEX_KDTREE = 1
# indexParams = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
# searchParams = dict(checks=50)

# flann = cv.FlannBasedMatcher(indexParams, searchParams)
# matches = flann.knnMatch(desStandard, desComparison, k=2)

# matchesMask = [[0, 0] for i in range(len(matches))]

# for i, (m, n) in enumerate(matches):
#     if m.distance < 0.7 * n.distance:
#         matchesMask[i] = [1, 0]

#     drawParams = dict(matchColor = (0, 255, 0), singlePointColor = (255, 0, 0), matchesMask = matchesMask, flags = 0)
#     imgResult = cv.drawMatchesKnn(imgStandard, kpStandard, imgComparison, kpComparison, matches, None, **drawParams)
#     plt.imshow(imgResult), plt.show()
##############################################################################

# imgGrayscale = cv.cvtColor(imgStandard, cv.COLOR_BGR2GRAY)    

# surf = cv.xfeatures2d.SURF_create(50000)

# kp, des = surf.detectAndCompute(imgGrayscale, None)

# print(len(kp))

# imgResult = cv.drawKeypoints(imgGrayscale, kp, None, (0, 0, 255), 4)

# cv.imshow('result', imgResult)
# cv.waitKey(0)