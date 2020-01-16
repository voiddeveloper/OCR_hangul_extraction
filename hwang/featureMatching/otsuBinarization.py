import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def thresholding():
    img = cv.imread('hwang/imgSet/test1.png', cv.IMREAD_GRAYSCALE)

    ret, thr1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

    ret, thr2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    blur = cv.GaussianBlur(img, (5,5), 0)
    ret, thr3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    titles = ['original', 'histogram', 'g-thresholding', 'orginal', 'histogram', 'otsu thresholding', 'gaussian', 'histogram', 'otsu thresholding']
    images = [img, 0, thr1, img, 0, thr2, blur, 0, thr3]

    for i in range(3):
        plt.subplot(3, 3, i*3+1), plt.imshow(images[i*3], 'gray')
        plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])

        plt.subplot(3,3, i*3+2), plt.hist(images[i*3].ravel(), 256)
        plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
        
        plt.subplot(3, 3, i*3+3), plt.imshow(images[i*3+2], 'gray')
        plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])

    plt.show()

# thresholding()