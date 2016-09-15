import PIL.Image as Image
import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt

grid = np.zeros(shape=(11,11,4,1,2), dtype=np.float32)
x0 = 260.0
y0 = 110.0
xi = x0
xif = x0
delta = 82
skip = 2
nCells = 11
frameLen = delta*nCells + (skip*(nCells-1))
x11 = x0 + frameLen
y11 = y0 + frameLen

xi  = x0
xif = x0
for i in range(0,11):
    xif = xif + delta
    xn  = int(xif)
    yj  = y0
    yjf = y0
    for j in range(0,11):
        yjf = yjf + delta
        yn = int(yjf)
        grid[i][j][0][0][0] = xi
        grid[i][j][0][0][1] = yj
        grid[i][j][1][0][0] = xn
        grid[i][j][1][0][1] = yj
        grid[i][j][2][0][0] = xn
        grid[i][j][2][0][1] = yn
        grid[i][j][3][0][0] = xi
        grid[i][j][3][0][1] = yn
        yj = yn + skip
    xi = xn + skip

grid2 = np.zeros(shape=(4,1,2), dtype='uint32')
grid2[0][0][0] = 226
grid2[0][0][1] = 408
grid2[1][0][0] = 1132
grid2[0][0][1] = 408
grid2[0][0][0] = 226
grid2[0][0][1] = 1314
grid2[1][0][0] = 1132
grid2[0][0][1] = 1314

# Initiate ORB detector with default values
SIFT = cv2.xfeatures2d.SIFT_create()

FLANN_INDEX_KDTREE = 0
INDEX_PARAMS = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
SEARCH_PARAMS = dict(checks = 50)
FLANN = cv2.FlannBasedMatcher(INDEX_PARAMS, SEARCH_PARAMS)

MIN_MATCH_COUNT = 8

def findCorner(cornerImg, keyPtC, descC, xRef, yRef, ticketImg, allGood):
    # find the keypoints and compute the descriptors with ORB
    keyPtT, descT = SIFT.detectAndCompute(ticketImg, None)
    matches = FLANN.knnMatch(descC,descT,2)
    print "FLANN found " + str(len(matches)) + " match candidates."
    # store all the good matches as per Lowe's ratio test.
    good = []
    any = []
    for m,n in matches:
        any.append(m)
        if m.distance < 0.7*n.distance:
            good.append(m)
            allGood.append(m)

    print "FLANN found " + str(len(good)) + " good match candidates."
    if len(good)>MIN_MATCH_COUNT:
        return matchesToPoint(cornerImg, keyPtC, keyPtT, xRef, yRef, ticketImg, good, True)
    # else:
        # return matchesToPoint(cornerImg, keyPtC, keyPtT, xRef, yRef, ticketImg, any, True)
    return 0, 0

def translatePoints(queryImg, keyPtQ, queryPoints, ticketImg. keyPtT, flannMatches, draw):
    # Find homography from query to target feature sets
    # Transform reference point of the query image to is corresponding location
    # relative to homologous region of query image in target image using the
    # alignment's perspective homography matrix.
    quryPts = np.float32([ keyPtQ[m.queryIdx].pt for m in flannMatches ]).reshape(-1,1,2)
    tcktPts = np.float32([ keyPtT[m.trainIdx].pt for m in flannMatches ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(quryPts, tcktPts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    tcktPoints = cv2.perspectiveTransform(queryPts,M)
    # Finally we draw our inliers (if successfully found the object) or matching
    # keypoints (if failed).
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
    imgOut = cv2.drawMatches(queryImg,keyPtQ,ticketImg,keyPtT,flannMatches,None,**draw_params)
    # cv2.imwrite(dataDir + '/' + matchFile + '.png', imgOut)
    # plt.imshow(imgOut, 'gray')
    # plt.show()
    print tcktCenter
    return (tcktCenter[0][0][0], tcktCenter[0][0][1])

# def mergeKeyPoints():
# 
# def findCells():
#         for i in range(0,11):
#             for j in range(0,11):
#                 print "ii=", i, "jj=", j
#                 cell = grid[i][j]
#                 cellTx = cv2.perspectiveTransform(cell,M)
#                 img2 = cv2.polylines(ticketImg, [np.int32(cellTx)], True, 255,3,cv2.LINE_AA)
#                 print str(grid[i][j][0][0][1]) + ":" + str(grid[i][j][3][0][1]) + ", " + str(grid[i][j][0][0][0]) + ":" + str(grid[i][j][1][0][0])
#                 cellImg = warpDstAsSrc[grid[i][j][0][0][1]:grid[i][j][3][0][1],grid[i][j][0][0][0]:grid[i][j][1][0][0]]
#                 cellFilePath = dataDir + '/' + cellFile + '_' +  str(i) + 'x' + str(j) + '.png'
#                 cv2.imwrite(cellFilePath, cellImg)
#                 print "Wrote cell file " + cellFilePath
#                 print cell
#                 print cellTx
#                 print cellImg
#     else:
#         print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
#         matchesMask = None
#     plt.imshow(warpDstAsSrc)
#     plt.show()
#     # Return in-memory output image to caller
#     return imgOut

# MAIN ROUTINE

# Read the four corner query images
tlCornerImg = cv2.imread('topLeft_v2.tif', 0)
trCornerImg = cv2.imread('topRght_v2.tif', 0)
blCornerImg = cv2.imread('btmLeft_v2.tif', 0)
brCornerImg = cv2.imread('btmRght_v2.tif', 0)

# Extract keypoints and descriptors for each of the four frame corners.
keyPtTLF, descrTLF = SIFT.detectAndCompute(tlCornerImg, None)
keyPtTRF, descrTRF = SIFT.detectAndCompute(trCornerImg, None)
keyPtBLF, descrBLF = SIFT.detectAndCompute(blCornerImg, None)
keyPtBRF, descrBRF = SIFT.detectAndCompute(brCornerImg, None)

dataDir = 'data'
ticketFile = 'ticket'
allGood = []

# Iterate over the training images
f = np.zeros(shape=(4,2), dtype='float32')
for ii in range(10,41):
    ticketFile = 'train/sample' + str(ii) + '.jpg'
    print "Reading " + ticketFile
    ticketImg = cv2.imread(ticketFile, 0)
    # find and draw the keypoints with and without supression
    print "Finding first corner"
    f[0][0],f[0][1] = findCorner(tlCornerImg, keyPtTLF, descrTLF, 216, 272, ticketImg, allGood)
    print "Finding second corner"
    f[1][0],f[1][1] = findCorner(trCornerImg, keyPtTRF, descrTRF, 436, 416, ticketImg, allGood)
    print "Finding third corner"
    f[2][0],f[2][1] = findCorner(brCornerImg, keyPtBRF, descrBRF, 606, 208, ticketImg, allGood)
    print "Finding fourth corner"
    f[3][0],f[3][1] = findCorner(blCornerImg, keyPtBLF, descrBLF, 224, 454, ticketImg, allGood)
    print "Found Grid: ", f
    img2 = cv2.polylines(ticketImg, [np.int32(f)], True, 255, 3, cv2.LINE_AA)
    plt.imshow(img2)
    plt.show()   

