import PIL.Image as Image
import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt

CELL_SCALE = 84
ROW_OFFSET = 4
COL_OFFSET = 2

# MAIN ROUTINE


def getImageBorderPath(sourceImg):
   (h,w) = sourceImg.shape
   retVal = np.array(shape=(4,1,2), dtype='float32')
   return getBorderPath(retVal, [0, 0], [w-1, 0], [w-1, h-1], [0, h-1])


def getCornerBorderPath(pathMatrix, lowLeft, lowRight, highRight, highLeft):
   pathMatrix[0][0] = [lowLeft[0], lowLeft[1]]
   pathMatrix[1][0] = [lowRight[0], lowRight[1]]
   pathMatrix[2][0] = [highRight[0], highRight[1]]
   pathMatrix[3][0] = [highLeft[0], highLeft[1]]
   return pathMatrix


cornerLocations = np.zeros(shape=(11,11,2), dtype='uint32')
gridLocation = np.zeros(shape=(4,1,2), dtype='uint32')
maxX = 0
maxY = 0
minX = 99999
minY = 99999

# Read the image and allocate buffers for two equivalently sized outputs
drawImg = cv2.imread('sample8.tif', 0)
pointSourceImg = cv2.imread('TopLeftGridPoints8.tif',0)
(numRows,numCols) = pointSourceImg.shape

for ii in range(300,numRows):
   for jj in range(100,numCols):
      if pointSourceImg[ii][jj] < 127:
         (x, y) = (jj, ii)
         xIdx = int(x/CELL_SCALE) - COL_OFFSET
         yIdx = int(y/CELL_SCALE) - ROW_OFFSET
         cornerLocations[yIdx][xIdx] = [y, x]
         if (x > maxX):
            maxX = x
         if (y > maxY):
            maxY = y
         if (x < minX):
            minX = x
         if (y < minY):
            minY = y
         print "%d at (%d,%d)" % (pointSourceImg[y][x], y, x)
         print "xIdx: %f->(%d), yIdx: %f->(%d)" % (x,xIdx,y,yIdx)
         print pointSourceImg[y:(y+4),x:(x+4)]
         pointSourceImg[y][x] = 255
         pointSourceImg[y][x+1] = 255
         pointSourceImg[y][x+2] = 255
         pointSourceImg[y][x+3] = 255
         pointSourceImg[y+1][x] = 255
         pointSourceImg[y+2][x] = 255
         pointSourceImg[y+3][x] = 255
         pointSourceImg[y+1][x+1] = 255
         pointSourceImg[y+1][x+2] = 255
         pointSourceImg[y+1][x+3] = 255
         pointSourceImg[y+2][x+1] = 255
         pointSourceImg[y+2][x+2] = 255
         pointSourceImg[y+2][x+3] = 255
         pointSourceImg[y+3][x+1] = 255
         pointSourceImg[y+3][x+2] = 255
         pointSourceImg[y+3][x+3] = 255

# Extrapolate length of 11 line segments from the 10 just measured.
maxX = maxX + ((maxX - minX)/10)
maxY = maxY + ((maxY - minY)/10)
getCornerBorderPath(gridLocation, [minY, minX], [maxY, minX], [maxY, maxX], [minY, maxX])

print gridLocation
print cornerLocations

knn = cv2.ml.KNearest_create()
trainData = cornerLocations.reshape(121,2).astype(np.float32)
knn.train(trainData, cv2.ml.ROW_SAMPLE, np.array(range(0,121)))
ret, results, neighbours, dist = knn.findNearest(trainData, 9)

print "ret: "
print ret
print "results: "
print results
print "neighbours: "
print neighbours
print "dist :"
print dist

print "minX=%f, maxX=%f, minY=%f, maxY=%f" % (minX,maxX,minY,maxY)

newGrid = []
imgOut1 = np.zeros(shape=(numRows,numCols), dtype='uint8')
# imgOut2 = np.zeros(shape=(numRows,numCols), dtype='uint8')
imgOut3 = drawImg.copy()
for ii in range(0,11):
  for jj in range(0,11):
    (y0, x0) = cornerLocations[ii][jj]
    print "%f, %f => %d, %d" % (xIdx, yIdx, x0, y0)

    imgOut1[y0-1][x0-1] = 255
    imgOut3[y0-1][x0-1] = 255
    imgOut1[y0][x0-1] = 255
    imgOut3[y0][x0-1] = 255
    imgOut1[y0+1][x0-1] = 255
    imgOut3[y0+1][x0-1] = 255
    imgOut1[y0-1][x0] = 255
    imgOut3[y0-1][x0] = 255

    imgOut1[y0][x0] = 255
    # imgOut2[y0][x0] = 255
    imgOut3[y0][x0] = 255

    imgOut1[y0+1][x0] = 255
    imgOut3[y0+1][x0] = 255
    imgOut1[y0-1][x0+1] = 255
    imgOut3[y0-1][x0+1] = 255
    imgOut1[y0][x0+1] = 255
    imgOut3[y0][x0+1] = 255
    imgOut1[y0+1][x0+1] = 255
    imgOut3[y0+1][x0+1] = 255

# print "Avg row dist: %f\nAvg col dist: %f\n" % (rowSum, colSum)

cv2.imwrite('sample8Points.tif', imgOut1)
cornerLocations.dump('sample8Corners.dat')
gridLocation.dump('sample8Grid.dat')

# plt.imshow(imgOut1, 'gray')
# plt.show()
# plt.imshow(imgOut2, 'gray')
# plt.show()
plt.imshow(imgOut3, 'gray')
plt.show()
