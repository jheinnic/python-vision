import PIL.Image as Image
import numpy as np
import cv2 as cv2
import os as os
from matplotlib import pyplot as plt

TICKET_MAX_FEATURES=16384
TICKET_SIFT_OCTAVES=1
TICKET_SIFT_SIGMA=1.825
TICKET_CANNY_THRESH_ONE=500
TICKET_CANNY_THRESH_TWO=500

CELL_MAX_FEATURES=64
CELL_SIFT_OCTAVES=3
CELL_SIFT_SIGMA=5.1735

# Initiate ORB detector with default values
FLANN_INDEX_KDTREE = 0
INDEX_PARAMS = dict(algorithm = FLANN_INDEX_KDTREE, trees = 8)
SEARCH_PARAMS = dict(checks = 80)

GRID_MIN_MATCH_PERCENT = 0.025
GRID_MAX_DIST_RATIO = 0.667

CELL_MIN_MATCH_PERCENT = 0.20
CELL_MAX_DIST_RATIO = 0.8

DATA_DIR = 'data22'
EXEMPLAR_TEMPLATE = 'letters10/%s%02d.tif'
CELL_LABEL_TEMPLATE = "(%d,%d) of %s"
CELL_MATCH_LABEL_TEMPLATE = "%s in %s"

CELL_OVERLAY_SOURCE_PCT=0.5
CELL_OVERLAY_EXAMPLE_PCT=1.0 - CELL_OVERLAY_SOURCE_PCT

NUM_CELLS=11
CELL_SIZE=96
CELL_CORRECTION=[-18,-18]

def fixedThreshold(imageArray, boundary):
   newArray = imageArray.copy()
   for eachRow in newArray:
      pixelCount = 0
      for eachPix in eachRow:
         if eachPix > boundary:
            eachRow[pixelCount] = 255
         else:
            if eachPix < boundary:
               eachRow[pixelCount] = 0
            else:
               eachRow[pixelCount] = 127
         pixelCount += 1
   return newArray

def avgThreshold(imageArray):
   pixelCount = 0
   balanceSum = 0
   for eachRow in imageArray:
      newRow = []
      for eachPix in eachRow:
         balanceSum += eachPix
         pixelCount += 1
   balance = balanceSum / pixelCount
   size = imageArray.size
   inOrder = np.sort(np.extract(imageArray, imageArray))
   print "Average pixel brightness is %f" % (balanceSum/pixelCount)
   newSize = inOrder.size
   p20Idx = newSize / 5
   p50Idx = newSize / 2
   print "numZeroes =", (size-newSize)
   print "Percentiles: 20th=>%d, 40th=>%d, 50th=>%d, 60th=>%d, 80th=>%d" % (inOrder[p20Idx], inOrder[p20Idx*2], inOrder[p50Idx], inOrder[p20Idx*3], inOrder[p20Idx*4])
   return fixedThreshold(imageArray, balance)

def pctThreshold(sourceImg, pctGoal):
   flat = sourceImg.reshape(-1)
   (counts, bounds) = np.histogram(flat, 96)
   total = len(flat)
   goal = pctGoal * total
   sum = 0
   for ii in range(0,96):
      sum += counts[ii]
      if sum > goal:
         goal = int(bounds[ii]) + 1
         print "%f brightness is %d" % (pctGoal, goal) 
         return fixedThreshold(sourceImg, goal)

def findPercentiles(sourceImg, targets):
   flat = sourceImg.reshape(-1)
   pixelCount = len(flat)
   numTargets = len(targets)
   nextBucketIdx = 0
   cumulativeSum = 0
   sortedTargets = np.sort(targets)
   (bucketCounts, bounds) = np.histogram(flat, 96)
   retVal = dict()
   for ii in range(0,numTargets):
      nextPctGoal = sortedTargets[ii]
      if nextPctGoal < 0 or nextPctGoal >= 1:
          print("Error!  Illegal percentile %f.  Legal values are greater than zero, but less than one." % (nextPctGoal))
          return None
      nextAbsGoal = nextPctGoal * pixelCount
      while cumulativeSum < nextAbsGoal:
         cumulativeSum += bucketCounts[nextBucketIdx]
         nextBucketIdx += 1
      threshold = int(bounds[nextBucketIdx-1])
      # print("%f brightness is at %d" % (nextPctGoal, threshold))
      retVal[nextPctGoal] = threshold
   return retVal

def cropAndResize(sourceImg, cropLowCorner, cropHighCorner, resizedX, resizedY):
   (xl,yl) = cropLowCorner
   (xh,yh) = cropHighCorner
   return cv2.resize(sourceImg[yl:yh,xl:xh].copy(), (resizedY, resizedX))

def getBorderPath(queryImg):
   (h,w) = queryImg.shape
   retVal = np.array([[[0.0, 0.0]],[[w-1.0,0.0]],[[w-1.0,h-1.0]],[[0.0,h-1.0]]])
   return retVal

def matchQueryToTrain(label, kpQ, desQ, kpT, desT, ratioTest, pctMinMatch):
   goodMatches = []
   # Shortcut this routine if we don't have enough descriptors to achieve the required number of matches.
   minMatch = pctMinMatch * len(kpQ)
   if kpT is None or desT is None:
      print "Skipping FLANN since there are no training descriptors!"
      return (False, None, None, goodMatches)
   if len(desT) < minMatch or len(desT) < minMatch:
      print "Skipping FLANN with only %d train descriptors" % (len(desT))
      return (False, None, None, goodMatches)
   print "Ready to FLANN with %d query and %d train descriptors" % (len(desQ), len(desT))
   flann = cv2.FlannBasedMatcher(INDEX_PARAMS, SEARCH_PARAMS)
   # store all the good matches as per Lowe's ratio test.
   for m,n in flann.knnMatch(desQ,desT,2):
      if m.distance <= (ratioTest * n.distance):
         goodMatches.append(m)
   if len(goodMatches)>=minMatch:
      print "Hit on %s with %d out of %d matches" % (label, len(goodMatches), minMatch)
      queryPts = np.float32([ kpQ[m.queryIdx].pt for m in goodMatches ]).reshape(-1,1,2)
      trainPts = np.float32([ kpT[m.trainIdx].pt for m in goodMatches ]).reshape(-1,1,2)
      return (True, queryPts, trainPts, goodMatches)
   else:
      print "Miss on %s with %d out of %d matches" % (label, len(goodMatches), minMatch)
      return (False, None, None, goodMatches)

def mapQueryToTrain(queryPts, trainPts, pointsInQuery):
   M, mask = cv2.findHomography(queryPts, trainPts, cv2.RANSAC,5.0)
   if M is not None:
      pointsInTrain = cv2.perspectiveTransform(pointsInQuery,M)
      return (M, mask.ravel().tolist(), pointsInTrain)
   return None, None, None

def warpTrainToQuery(queryPts, queryImg, trainPts, trainImg):
   # Produce a "copy" of the query image by warped extraction from the training image.
   iM, iMask = cv2.findHomography(trainPts, queryPts, cv2.RANSAC, 5.0)
   return (iM, iMask.ravel().tolist(), cv2.warpPerspective(trainImg, iM, (len(queryImg[0]), len(queryImg))))

class SiftImage:
   def __init__(self, label, selfImg, filePath, siftInst):
      self.label = label
      self.selfImg = selfImg
      if filePath is not None:
         cv2.imwrite(filePath, selfImg)
      (self.keypoints, self.descriptors) = siftInst.detectAndCompute(selfImg, None)
      self.borderPath = getBorderPath(selfImg)

   def getFeatureCount(self):
      if self.descriptors is None:
         return 0
      return len(self.descriptors)

   def getKeypoints(self):
      return self.keypoints

   def getDescriptors(self):
      return self.descriptors

   def detectAndCompute(selfself):
      return (self.keypoints, self.descriptors)

   def getDetails(self):
      return (self.selfImg, self.keypoints, self.descriptors, self.borderPath)

   def getBorderPath(self):
      return self.borderPath

   def getImage(self):
      return self.selfImg

   def getLabel(self):
      return self.label


class LetterMatcher:
   def __init__(self, letter, siftInst):
       exemplarFile = EXEMPLAR_TEMPLATE % (letter, 1)
       exemplarImg = cv2.imread(EXEMPLAR_TEMPLATE % (letter, 1), 0)
       print exemplarFile
       self.siftImage = SiftImage(letter, exemplarImg, None, siftInst)
   
   def getDetails(self):
      return self.siftImage.getDetails()

   def getBorderPath(self):
      return self.siftImage.getBorderPath()

   def getLabel(self):
      return self.siftImage.getLabel()


class TicketCell:
   def __init__(self, cellImg, cellSifter, cellLabel, cellFilePath, matchFileTemplate):
      self.cellImg = SiftImage(cellLabel, cellImg, cellFilePath, cellSifter)
      self.matchFileTemplate = matchFileTemplate
      self.candidateHits = {}
      percentiles = findPercentiles(cellImg, [0.15, 0.3])
      print('Cell is blank if (%d+4) >= %d' % (percentiles[0.15], percentiles[0.3]))
      if (percentiles[0.15] + 9) < percentiles[0.3]:
         self.contentString = ''
      else:
         self.contentString = '-'

   def visualizeByOverlay(self, cellImg, exemplarImg, resultRaw, topLeft, letter, fileTemplate):
      (lettH,lettW) = exemplarImg.shape
      (cellH,cellW) = cellImg.shape
      cellLettH = min(lettH, cellH-topLeft[1])
      cellLettW = min(lettW, cellW-topLeft[0])
      btmRight = (topLeft[0]+cellLettW, topLeft[1]+cellLettH)
      mask = np.zeros((cellH,cellW,3), dtype='uint8')
      source = np.zeros((cellH,cellW,3), dtype='uint8')
      for ii in range(topLeft[1],btmRight[1]):
         for jj in range(topLeft[0],btmRight[0]):
            if exemplarImg[ii-topLeft[1]] < 128:
                mask[ii][jj][0] = 255
            else:
                mask[ii][jj][1] = 255
      for ii in range(0, cellH):
         for jj in range(0, cellW):
            source[ii][jj][2] = cellImg[ii][jj]
      mask[topLeft[1]:btmRight[1],topLeft[0]:btmRight[0]] = exemplarImg
      maskedCellImg = cv2.addWeighted(source, CELL_OVERLAY_SOURCE_PCT, mask, CELL_OVERLAY_EXAMPLE_PCT, 0)
      cv2.imwrite(fileTemplate % ('overlay', letter), maskedCellImg)
      # Normalized visualiations of match
      result8 = cv2.normalize(resultRaw, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
      cv2.imwrite(self.matchFileTemplate % ('normScore', letter), result8)

   def identify(self, matchLog):
      # Shortcut if we already determied no content eixsts here.
      if self.contentString == '-':
         return '-'
      letterIdx = 0
      maxScore = 0
      maxCoord = None
      (cellImg, kpC, desC, cellBorder) = self.cellImg.getDetails()
      cellLabel = self.cellImg.getLabel()
      # Now begin checking for a classification hit
      for letter in alphaSplit:
         letterObj = exemplars[letter]
         (exemplarImg, kpE, desE, exemplarBorder) = letterObj.getDetails()
         matchLabel = CELL_MATCH_LABEL_TEMPLATE % (letter, cellLabel)
         resultRaw = cv2.matchTemplate(cellImg, exemplarImg, cv2.TM_CCOEFF)
         cv2.imwrite(self.matchFileTemplate % ('rawMatch', letter), resultRaw)
         # resultCubed = resultRaw**3
         # val, resultScore = cv2.threshold(resultCubed, 0.01, 0, cv2.THRESH_TOZERO)
         # Track the high score
         (lowScore, highScore, minLoc, maxLoc) = cv2.minMaxLoc(resultRaw)
         topLeft = maxLoc
         if (highScore > maxScore):
            maxCoord = topLeft
            maxScore = highScore
            self.contentString = letter
            print("New best score for %s at (x=%d,y=%d) of %s.  Score=%f" % (letter, topLeft[1], topLeft[0], self.cellImg.getLabel(), highScore))
            matchLog.write('%s|raw|high|%s|%d|%d|%f|%s\n' % (self.cellImg.getLabel(), letter, topLeft[1], topLeft[0], highScore, self.contentString))
         else:
            if highScore == maxScore:
               # Track all best letters, but only the location of the first to find
               self.contentString = self.contentString + letter
               print("Tied best score for %s at (x=%d,y=%d) of %s.  Score=%f" % (letter, topLeft[1], topLeft[0], self.cellImg.getLabel(), highScore))
               matchLog.write('%s|raw|tie|%s|%d|%d|%f|%s\n' % (self.cellImg.getLabel(), letter, topLeft[1], topLeft[0], highScore, self.contentString))
            else:
               print("Low score for %s at (x=%d,y=%d) of %s.  Score=%f" % (letter, topLeft[1], topLeft[0], self.cellImg.getLabel(), highScore))
               matchLog.write('%s|raw|low|%s|%d|%d|%f|%s\n' % (self.cellImg.getLabel(), letter, topLeft[1], topLeft[0], highScore, self.contentString))
         # Visualize each match--high score or not
         self.visualizeByOverlay(cellImg, exemplarImg, resultRaw, topLeft, letter, self.matchFileTemplate)
         # Advance the current letter index to next value.
         letterIdx += 1
      # If no matches were found, treat the content same as a black blocked square by assigning the
      # same content value used in that scenario.
      if self.contentString == '':
         self.contentString = '-'
         print("No identifying content found in %s" % (self.cellImg.getLabel()))
      return self.contentString

#   def showMulti(resultImages):
#          figRows = 4
#          figCols = 7
#          figRowSpan = 4
#          figColSpan = 5
#          figRowGap = 0
#          figColGap = 0
#          figRowMulti = figRowSpan + figRowGap
#          figColMulti = figColSpan + figColGap
#          yyMax = (figRows * figRowMulti) - figRowGap
#          xxMax = (figCols * figColMulti) - figColGap
#          axes = np.ndarray(shape=(27), dtype='object')
#          fig = plt.figure()
#          for ii in range(0,26):
#              yy = ii / figCols
#              xx = ii % figCols
#              yyPlot = figRowMulti * yy
#              xxPlot = figColMulti * xx
#              if (yy >= 3):
#                 xxPlot = xxPlot + figRowMulti
#              axes[ii] = plt.subplot2grid((yyMax,xxMax), (yyPlot,xxPlot), rowspan=figRowSpan, colspan=figColSpan)
#              # print("(%d,%d)" % (xxPlot,yyPlot))
#          for ii in range(0,4):
#             print("Rendering image %d of 5" % (ii+1))
#             for jj in range(0,26):
#                # print("ii=%d, jj=%d" % (ii, jj))
#                # print(resultImages[ii][jj])
#                axes[jj].imshow(resultImages[ii][jj])
#             # key = plt.waitKey(0)
#             # print(key)
#             plt.show(block=False)
#             print "Leaving render"

class CrosswordTicket:
   ticketSIFT = cv2.xfeatures2d.SIFT_create(TICKET_MAX_FEATURES, TICKET_SIFT_OCTAVES, 0, 0, TICKET_SIFT_SIGMA)
   
   # Read the image and allocate buffers for two equivalently sized outputs
   cornerLocations = np.load('sample8Corners3.dat')
   for ii in range(0,NUM_CELLS):
      for jj in range(0,NUM_CELLS):
         cornerLocations[ii][jj] = cornerLocations[ii][jj] + CELL_CORRECTION

   gridLocation = np.load('sample8Grid.dat')
   top = max(gridLocation[1][0][1], gridLocation[2][0][1]) + 1
   bottom = min(gridLocation[0][0][1], gridLocation[3][0][1])
   right = max(gridLocation[2][0][0], gridLocation[3][0][0]) + 1
   left = min(gridLocation[0][0][0], gridLocation[1][0][0])
   print "%d:%d,%d:%d" % (bottom, top, left, right)
   gridIndex = np.ogrid[bottom:top,left:right]

   def __init__(self, ticketFilePath):
      ticketImg = cv2.imread(ticketFilePath, 0)
      if ticketImg is None:
         print('Could not open %s' % (ticketFilePath))
         return IOError() 
      self.ticketImg = SiftImage(ticketFilePath, ticketImg, None, CrosswordTicket.ticketSIFT)
      self.M = None
      self.iM = None
      self.matchMask = None
      self.queryObj = None
      self.goodMatches = None
      self.warpedImg = None
         
   def locateCrossword(self, queryObj):
       kpQ = queryObj.getKeypoints()
       desQ = queryObj.getDescriptors()
       kpT = self.ticketImg.getKeypoints()
       desT = self.ticketImg.getDescriptors()
       (found, self.framePts, self.ticketPts, self.goodMatches) = matchQueryToTrain('ticket', kpQ, desQ, kpT, desT, GRID_MAX_DIST_RATIO, GRID_MIN_MATCH_PERCENT)
       if found == True:
          (self.M, self.matchesMask) = cv2.findHomography(self.framePts, self.ticketPts, cv2.RANSAC, 5.0)
          if self.M is None:
             print "## No homography found for derived matching from query template to %s" % (ticketFile)
             return False
          # Warp so we can use cell locations using coordinates from the query image to find letters.
          self.queryObj = queryObj
          (_, self.iM) = cv2.invert(self.M)
          # print self.M
          # print self.iM
          ticketImg = self.ticketImg.getImage()
          (numRows,numCols) = queryObj.getImage().shape
          self.warpedImg = cv2.warpPerspective(ticketImg, self.iM, (numCols, numRows))
       return found

   def visualizeCrosswordExtraction(self, sampleDir):
      if self.warpedImg is None:
          print "Cannot visualize crossword extract until its location has been found!"
          return False
      # Extract a crop of just the target area for debug purposes
      extractedImg = self.warpedImg[CrosswordTicket.gridIndex]
      print extractedImg
      print sampleDir + '/warpGridCrop.tif'
      cv2.imwrite(sampleDir + '/warpGridCrop.tif', extractedImg)
      return True


   def visualizeCrosswordLocation(self, sampleDir):
      if self.M is None:
           print "Cannot visualize location until that location has been found!"
           return False
      M = self.M
      kpT = self.ticketImg.getKeypoints()
      kpQ = self.queryObj.getKeypoints()
      queryImg = self.queryObj.getImage()
      ticketImg = self.ticketImg.getImage()
      foundInRegion = np.int32(cv2.perspectiveTransform(np.float32(CrosswordTicket.gridLocation), M))
      decoratedImg = ticketImg.copy()
      # print decoratedImg.dtype
      decoratedImg = cv2.drawKeypoints(decoratedImg, kpT, None, color=(192,128,0)) #, flags=0)
      decoratedImg = cv2.polylines(decoratedImg, [foundInRegion], True, 192, 5, cv2.LINE_AA)
      cv2.imwrite(sampleDir + '/warpFullTicket.tif', decoratedImg)
      print decoratedImg.shape
      # Create side-by-side visualization of the perspective transformation's source match support.
      draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                         singlePointColor = (0,0,255), # draw unmatched keyponts in blue?
                         matchesMask = self.matchesMask.ravel().tolist(), # draw only inliers
                         flags = 2)
      flannMatchImg = cv2.drawMatches(queryImg, kpQ, decoratedImg, kpT, self.goodMatches, None, **draw_params)
      cv2.imwrite(sampleDir + '/flannWarp.tif', flannMatchImg)
      return True
 
   def processCrosswordGrid(self, sampleDataDir):
      if self.warpedImg is None:
          print "Cannot visualize crossword extract until its location has been found!"
          return None
      cornerLocations = CrosswordTicket.cornerLocations
      cellFileTemplate = sampleDataDir + '/cell_%dx%d.tif'
      matchLog = file( sampleDataDir + '/templateMatchScores.log', 'w')
      cellSifter = cv2.xfeatures2d.SIFT_create(CELL_MAX_FEATURES, CELL_SIFT_OCTAVES, 0, 0, CELL_SIFT_SIGMA)
      xwGrid = np.chararray(shape=(NUM_CELLS,NUM_CELLS), itemsize=20)
      xwGrid.fill('-')
      xwStr = ''
      for ii in range(0,NUM_CELLS):
         for jj in range(0,NUM_CELLS):
            (bottom,left) = cornerLocations[ii][jj]
            (top,right) = cornerLocations[ii][jj] + (CELL_SIZE,CELL_SIZE)
            print "%d:%d,%d:%d" % (bottom, top, left, right)
            cellImg = self.warpedImg[bottom:top,left:right]
            cellLabel = CELL_LABEL_TEMPLATE % (ii, jj, self.ticketImg.getLabel())
            cellCoordStr = '%dx%d' % (ii, jj)
            cellFilePath = cellFileTemplate % (ii,jj)
            matchFileTemplate = sampleDataDir + '/match-%s_' + cellCoordStr + '_%s.tif'
            cellObject = TicketCell(cellImg, cellSifter, cellLabel, cellFilePath,  matchFileTemplate)
            xwGrid[ii][jj] = cellObject.identify(matchLog)
            xwStr += xwGrid[ii][jj] + ' '
         xwStr += "\n"
      print xwStr
      output = file( sampleDataDir + '/xwGrid.txt', 'w')
      output.write(xwStr)
      output.close()
      matchLog.close()
      xwGrid.dump(sampleDataDir + '/xwGrid.dat')
      return xwGrid

###
# MAIN ROUTINE
###

print "Loading..."

# Read the query image and find its keypoints and their descriptors with ORB
queryImg = cv2.imread('sample8.tif', 0)
queryObj = SiftImage('ticketQuery', queryImg, None, CrosswordTicket.ticketSIFT)

exemplars = { }
alphas = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z"
alphaSplit = alphas.split(' ')
eSIFT = cv2.xfeatures2d.SIFT_create(CELL_MAX_FEATURES, CELL_SIFT_OCTAVES, 0, 0, CELL_SIFT_SIGMA)
for letter in alphaSplit:
   exemplars[letter] = LetterMatcher(letter, eSIFT)

# Iterate over the training images
for ii in range(16,19):
   # Construct next file name and search it
   ticketFile = 'train5/sample%02d.tif' % (ii)
   ticketObj = CrosswordTicket(ticketFile)
   print "Ingesting " + ticketFile
   if ticketObj.locateCrossword(queryObj):
      sampleDir = '%s/sample%02d' % (DATA_DIR, ii)
      try:
         os.makedirs(sampleDir, 0755)
      except:
         print('%s already exists' % (sampleDir))
      # Save images of the matching and extracted crossword grid
      ticketObj.visualizeCrosswordExtraction(sampleDir)
      ticketObj.visualizeCrosswordLocation(sampleDir)
      # Analyze the crossword and interpret its content!
      ticketObj.processCrosswordGrid(sampleDir) 
      print "** Finished crossword ingestion from %s" % (ticketFile)
   else:
      print "## No feature matches found between query template and %s" % (ticketFile)

