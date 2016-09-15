import PIL.Image as Image
import numpy as np
import cv2 as cv2

TEST_SIZE = (48,60)
RAND_DELTA = (TEST_SIZE[0]/6,TEST_SIZE[1]/6)
FILE_TEMPLATE = 'antiSamples/%04d.tif'
RANDOM_PHASE_COUNT = 4200

def findBottomRight(pt):
   return [pt[0]+TEST_SIZE[0],pt[1]+TEST_SIZE[1]]

def takeClip(img, pt, coverage, nextIdx):
   pt2 = findBottomRight(pt)
   print pt, ' -> ', pt2
   clip = img[pt[0]:pt2[0],pt[1]:pt2[1]]
   file = FILE_TEMPLATE % (nextIdx)
   cv2.imwrite(file, clip)
   coverage[pt[0]:pt2[0],pt[1]:pt2[1]] += 1
   return nextIdx + 1

# Apply a random offset to input (x,y) point and return the derived result as a (y,x) point
def applyRandomOffset(xyPt,domainCorner):
   np.random.seed()
   deltaY = np.random.randint(1, RAND_DELTA[0], (1))
   deltaX = np.random.randint(1, RAND_DELTA[1], (1))
   return [min(domainCorner[0],max(0,xyPt[1]-deltaY[0])), min(domainCorner[1],max(0,xyPt[0]-deltaX[0]))]

def generateRandomPoints(pointCount, domainShape):
   np.random.seed()
   randomY = np.random.randint(0,domainShape[0],(pointCount,1))
   randomX = np.random.randint(0,domainShape[1],(pointCount,1))
   random = np.ndarray((pointCount,2), dtype='int')
   random[:pointCount,0:1] = randomY
   random[:pointCount,1:2] = randomX
   return random.tolist()

def checkpoint(pointList, fileName):
   dump = np.zeros((len(pointList),2))
   dump[:] = pointList
   dump.dump(fileName)

def coverPointAt(minVal, xyPt, domainCorner, pointList, nextIdx):
    yxPt = applyRandomOffset(xyPt, domainCorner)
    print('Adding a sample at (%d,%d) to cover minVal=%d at (%d,%d)' % (yxPt[1], yxPt[0], minVal, xyPt[0], xyPt[1]))
    pointList.append(yxPt)
    return takeClip(img, yxPt, coverage, nextIdx)


img = cv2.imread('blankTicket.tif', 0)
imgShape = img.shape
domainCorner = (imgShape[0]-TEST_SIZE[0], imgShape[1]-TEST_SIZE[1])
domainShape = (domainCorner[0]+1, domainCorner[1]+1)

clip = img[domainCorner[0]:imgShape[0],domainCorner[1]:imgShape[1]]
print clip.shape

random = generateRandomPoints(RANDOM_PHASE_COUNT, domainShape)
coverage = np.zeros(imgShape)
boundaries = np.zeros((4))
boundaries[0] = len(random) - 1
nextIdx = 0
nextIdx = takeClip(img, [domainCorner[0],domainCorner[1]], coverage, nextIdx)
nextIdx = takeClip(img, [0, 0], coverage, nextIdx)

for yxPt in random:
    nextIdx = takeClip(img, yxPt, coverage, nextIdx)

(minVal, maxVal, xyPt, maxPt) = cv2.minMaxLoc(coverage)
coverage.dump('initialCoverage.dat')
checkpoint(random, 'initialPoints.dat')

while minVal < 1:
    nextIdx = coverPointAt(minVal, xyPt, domainCorner, random, nextIdx)
    (minVal, maxVal, xyPt, maxPt) = cv2.minMaxLoc(coverage)

print('Minimal coverage achieved by nextIdx=%d.  Minval=%d at (%d,%d)' % (nextIdx-1, minVal, xyPt[1], xyPt[0]))
coverage.dump('oneCoverge.dat')
checkpoint(random, 'onePoints.dat')
boundaries[1] = nextIdx - 1

while minVal < 2:
    nextIdx = coverPointAt(minVal, xyPt, domainCorner, random, nextIdx)
    (minVal, maxVal, xyPt, maxPt) = cv2.minMaxLoc(coverage)
print('Two-fold coverage achieved by nextIdx=%d.  Minval=%d at (%d,%d)' % (nextIdx-1, minVal, xyPt[1], xyPt[0]))
coverage.dump('twoCoverage.dat')
checkpoint(random, 'twoPoints.dat')
boundaries[2] = nextIdx - 1

while minVal < 3:
    nextIdx = coverPointAt(minVal, xyPt, domainCorner, random, nextIdx)
    (minVal, maxVal, xyPt, maxPt) = cv2.minMaxLoc(coverage)
print('Two-fold coverage achieved by nextIdx=%d.  Minval=%d at (%d,%d)' % (nextIdx-1, minVal, xyPt[1], xyPt[0]))
coverage.dump('triCover.dat')
checkpoint(random, 'triPoints.dat')
boundaries[3] = nextIdx - 1

print "Boundaries summary: ", boundaries
boundaries.dump('boundaries.dat')
output = file('boundaries.txt', 'w')
for ii in range(0,4):
   output.write(str(boundaries[ii]) + "\n")

output.close()
