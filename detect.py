import cv2 as cv

img = cv.imread('./data/175806.png')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print(img.shape)
h, w = img.shape
factor = max(224 / h, 224 / w)
img = cv.resize(img, None, fx = factor, fy = factor)
print(img.shape)
edge = cv.Canny(img, 100, 200)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
img = cv.morphologyEx(edge, cv.MORPH_CLOSE, kernel)
cv.imshow('', img)
cv.waitKey(0)