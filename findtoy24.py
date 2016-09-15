import PIL.Image as Image
import numpy as np
import cv2 as cv2
import os as os
from matplotlib import pyplot as plt

TICKET_MAX_FEATURES=16384
TICKET_SIFT_OCTAVES=1
TICKET_SIFT_SIGMA=1.825

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

DATA_DIR = 'data24'
EXEMPLAR_FILE_TEMPLATE = 'letters10/%s%02d.tif'
CELL_LABEL_TEMPLATE = "(%d,%d) of %s"
MATCH_LABEL_TEMPLATE = "%s in %s"

CELL_OVERLAY_SOURCE_PCT=0.65
CELL_OVERLAY_EXAMPLE_PCT=1.0 - CELL_OVERLAY_SOURCE_PCT

NUM_CELLS=11
CELL_SIZE=96
CELL_CORRECTION=[-18,-18]
BLANK_SIZE=64
BLANK_CORRECTION=[16,16]

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

# class FlannLocator:
#    flann = cv2.FlannBasedMatcher(INDEX_PARAMS, SEARCH_PARAMS)
# 
#    def __init__(self, matchLabel, querySiftImg, targetSiftImg, ratioTest, pctMinMatch):
#       # Shortcut this routine if we don't have any keypoints to consider from either image.
#       numFeaturesQ = querySiftImage.getFeatureCount( )
#       if numFeaturesQ <= 0:
#          print "FLANN matching requires query keypoints/descriptors!"
#          return (False, None, None, goodMatches)
#       numFeaturesT = targetSiftImage.getFeatureCount( )
#       if numFeaturesT <= 0:
#          print "FLANN matching requires target keypoints/descriptors!"
#          return (False, None, None, goodMatches)
#       self.minMatch = pctMinMatch * len(kpQ)
#       if numFeaturesT < (numFeaturesQ * pctMinMatch):
#          print('%d target features is insufficient for the %d required to match with %d query features' % (numFeaturesT, (numFeaturesQ * pctMinMatch), numFeaturesQ))
#          return (False, None, None, [])
# 
#       print "FLANN will match %d query to %d candidate keypoints/descriptors" % (len(desQ), len(desT))
#       self.matchLabel = matchLabel
#       self.querySiftImg = querySiftImg
#       self.targetSiftImg = targetSiftImg
#       # store all the good matches as per Lowe's ratio test.
#       self.goodMatches = []
#       for m,n in FlannLocator.flann.knnMatch(desQ,desT,2):
#          if m.distance <= (ratioTest * n.distance):
#             self.goodMatches.append(m)
#       if len(self.goodMatches) >= self.minMatch:
#          print "Hit on %s with %d out of %d matches" % (label, len(self.goodMatches), self.minMatch)
#          self.queryPts = np.float32([ kpQ[m.queryIdx].pt for m in goodMatches ]).reshape(-1,1,2)
#          self.trainPts = np.float32([ kpT[m.trainIdx].pt for m in goodMatches ]).reshape(-1,1,2)
#       else:
#          print "Miss on %s with %d out of %d matches" % (label, len(self.goodMatches), self.minMatch)
#          self.queryPts = None
#          self.trainPts = None
#       self.matchMatrix = None
#       self.inverseMatchMatix = None
#       if len(self.goodMatches) >= self.minMatch:
#          return (True, self.queryPts, self.trainPts, self.goodMatches)
#       else:
#          return (False, None, None, [])

def matchQueryToTrain(label, kpQ, desQ, kpT, desT, ratioTest, pctMinMatch):
   goodMatches = []
   # Shortcut this routine if we don't have any keypoints to consider from ticket image.
   if kpT is None or desT is None:
      print "Skipping FLANN since there are no training descriptors!"
      return (False, None, None, goodMatches)
   # Shortcut this routine if we don't have enough descriptors to achieve the required number of matches.
   minMatch = max((pctMinMatch * min(len(kpQ), len(kpT))), 10)
   minMatch = 20
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



class SiftImage:
   def __init__(self, label, selfImg, siftInst):
      self.label = label
      self.selfImg = selfImg
      (self.keypoints, self.descriptors) = siftInst.detectAndCompute(self.selfImg, None)
      if len(self.keypoints) != len(self.descriptors):
          print 'TODO: fail with error'
      if len(self.keypoints) <= 0:
          print 'TODO: fail with error'
      if len(self.descriptors) <= 0:
          print 'TODO: fail with error'


   def getFeatureCount(self):
      if self.descriptors is None:
         return 0
      return len(self.descriptors)

   def getKeypoints(self):
      return self.keypoints

   def getDescriptors(self):
      return self.descriptors

   def getImage(self):
      return self.selfImg

   def getLabel(self):
      return self.label

   def getFeatures(self):
      return (self.keypoints, self.descriptors)

   def getAllDetails(self):
      return (self.label, self.selfImg, self.keypoints, self.descriptors)


class LetterMatcher:
   def __init__(self, letter):
      self.letter = letter
      self.tiedWith = None
      self.bestScore = None
      self.bestCoords = None
      self.matchLabel = None
      self.candidateImg = None
      self.exemplarFile = EXEMPLAR_FILE_TEMPLATE % (letter, 1)
      self.exemplarImg = cv2.imread(self.exemplarFile, 0)

   def scoreCandidateMatch(self, sourceLabel, candidateImg):
      self.candidateImg = candidateImg
      self.matchLabel = MATCH_LABEL_TEMPLATE % (self.letter, sourceLabel)
      self.tiedWith = ''
      rawResult = cv2.matchTemplate(candidateImg, self.exemplarImg, cv2.TM_CCOEFF)
      (_, self.bestScore, _, self.bestCoords) = cv2.minMaxLoc(rawResult)
      print('<%f> at <%d,%d>' % (self.bestScore, self.bestCoords[1], self.bestCoords[0]))
      return (self.bestScore, self.bestCoords)

   # Visualize each match--high score or not
   def withCandidate(self, cellLabel, cellImg, matchFileTemplate):
      self.scoreCandidateMatch(cellLabel, cellImg)
      self.visualizeByOverlay(matchFileTemplate % ('overlay', self.letter))
      return self
  
   # Return True if self has the better score, False if other has the better score, and None if the
   # scores are tied.  
   def compareTo(self, other):
      if self.bestScore > other.bestScore:
         self.tiedWith = ''
         return True
      if self.bestScore < other.bestScore:
         return False
      # Special case handling for ties
      tiedWithOther = self.letter + self.tiedWith
      tiedWithSelf  = other.letter + other.tiedWith
      self.tiedWith += tiedWithSelf
      other.tiedWith += tiedWithOther
      return None

   def visualizeByOverlay(self, outputFilePath):
      topLeft = self.bestCoords
      letterImg = self.exemplarImg
      cellImg = self.candidateImg
      (lettH,lettW) = letterImg.shape
      (cellH,cellW) = cellImg.shape
      # Truncate any overflow beyond the boundary if a match's top left corner is too close to candidate
      # patch's bottom and/or right-hand edges.
      cellLettH = min(lettH, cellH-topLeft[1])
      cellLettW = min(lettW, cellW-topLeft[0])
      btmRight = (topLeft[0]+cellLettW, topLeft[1]+cellLettH)
      # Allocate colorspace array to hold the visualization as its computed
      mask = np.zeros((cellH,cellW,3), dtype='uint8')
      for ii in range(topLeft[1],btmRight[1]):
         for jj in range(topLeft[0],btmRight[0]):
            mask[ii][jj][0] = cellImg[ii][jj]
            sigma = cellImg[ii][jj] / 4
            if letterImg[ii-topLeft[1]][jj-topLeft[0]] < 128:
                mask[ii][jj][1] = sigma
                mask[ii][jj][2] = 255 - sigma 
            else:
                sigma = 64 - sigma
                mask[ii][jj][1] = 255 - sigma
                mask[ii][jj][2] = sigma
      for ii in range(0, cellH):
         for jj in range(0, topLeft[0]):
            mask[ii][jj][0] = cellImg[ii][jj]
            mask[ii][jj][1] = mask[ii][jj][2] = (cellImg[ii][jj] / 2)
            mask[ii][jj][1] = mask[ii][jj][2] = (cellImg[ii][jj] / 2)
      for jj in range(topLeft[0],btmRight[0]):
         for ii in range(0, topLeft[1]):
            mask[ii][jj][0] = cellImg[ii][jj]
            mask[ii][jj][1] = mask[ii][jj][2] = (cellImg[ii][jj] / 2)
         for ii in range(btmRight[0], cellW):
            mask[ii][jj][0] = cellImg[ii][jj]
            mask[ii][jj][1] = mask[ii][jj][2] = (cellImg[ii][jj] / 2)
      # maskedCellImg = cv2.addWeighted(mask, CELL_OVERLAY_SOURCE_PCT, mask, CELL_OVERLAY_EXAMPLE_PCT, 0)
      # cv2.imwrite(matchFileTemplate % ('overlay', letter), mask)
      cv2.imwrite(outputFilePath, mask)
      # Normalized visualiations of match
      # result8 = cv2.normalize(resultRaw, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
      # cv2.imwrite(self.matchFileTemplate % ('normScore', letter), result8)


class TicketCell:
   def __init__(self, cellImg, cellLabel, ticketLabel, matchFileTemplate):
      self.cellImg = cellImg 
      print "cellLabel"
      self.cellLabel = cellLabel 
      self.ticketLabel = ticketLabel 
      self.matchFileTemplate = matchFileTemplate
      self.contentString = None

   def exportCell(self, filePath):
       if filePath is not None and self.cellImg is not None:
         cv2.imwrite(filePath, selfImg)

   def identifyContent(self, exemplars, matchLog):
      # Shortcut if we already predict no content.  A Cell with content has a significant difference in
      # the luminosity of pixels at the 15th an 30th percentiles.  A black block with no content, on the
      # other hand, has negligible difference between pixels at the same percentile range.
      blankH = BLANK_CORRECTION[0]
      blankW = BLANK_CORRECTION[1]
      blankTestImg = cellImg[blankH:(blankH+BLANK_SIZE),blankW:(blankW+BLANK_SIZE)]
      percentiles = findPercentiles(blankTestImg, [0.20, 0.33])
      if (percentiles[0.20] + 9) >= percentiles[0.33]:
         self.contentString = '-'
         return '-'
      # Now search for the best match by finding the one with the best score.
      endPoint = 0
      searchOrder = np.ndarray(shape=(26), dtype='object')
      for letter in exemplars.itervalues():
         searchOrder[endPoint] = letter.withCandidate(self.cellLabel, self.cellImg, self.matchFileTemplate)
         endPoint += 1
      while endPoint > 1:
          halfPoint = endPoint / 2
          for ii in range(0, halfPoint):
             endPoint = endPoint - 1
             letterOne = searchOrder[ii]
             letterTwo = searchOrder[endPoint]
             oneWins = letterOne.compareTo(letterTwo)
             if oneWins is None:
                # In case of a tie letterOne references letterTwo via its "tiedWith" property and can
                # therefore be treated as a winner here now.
                print('At %s, %s tied %s with %f at (x=%d, y=%d) versus %f at (x=%d, y=%d)' % (self.cellLabel, letterOne.letter, letterTwo.letter, letterOne.bestScore, letterOne.bestCoords[1], letterOne.bestCoords[0], letterTwo.bestScore, letterTwo.bestCoords[1], letterTwo.bestCoords[0]))
                matchLog.write('%s|%s|%d|%d|%f|tie\n' % (self.cellLabel, letterOne.letter, letterOne.bestCoords[1], letterOne.bestCoords[0], letterOne.bestScore))
                matchLog.write('%s|%s|%d|%d|%f|tie\n' % (self.cellLabel, letterTwo.letter, letterTwo.bestCoords[1], letterTwo.bestCoords[0], letterTwo.bestScore))
             if oneWins == True:
                print('At %s, %s defeats %s with %f at (x=%d, y=%d) versus %f at (x=%d, y=%d)' % (self.cellLabel, letterOne.letter, letterTwo.letter, letterOne.bestScore, letterOne.bestCoords[1], letterOne.bestCoords[0], letterTwo.bestScore, letterTwo.bestCoords[1], letterTwo.bestCoords[0]))
                matchLog.write('%s|%s|%d|%d|%f|win\n' % (self.cellLabel, letterOne.letter, letterOne.bestCoords[1], letterOne.bestCoords[0], letterOne.bestScore))
                matchLog.write('%s|%s|%d|%d|%f|loss\n' % (self.cellLabel, letterTwo.letter, letterTwo.bestCoords[1], letterTwo.bestCoords[0], letterTwo.bestScore))
             if oneWins == False:
                print('At %s, %s is defeated by %s with %f at (x=%d, y=%d) versus %f at (x=%d, y=%d)' % (self.cellLabel, letterOne.letter, letterTwo.letter, letterOne.bestScore, letterOne.bestCoords[1], letterOne.bestCoords[0], letterTwo.bestScore, letterTwo.bestCoords[1], letterTwo.bestCoords[0]))
                matchLog.write('%s|%s|%d|%d|%f|loss\n' % (self.cellLabel, letterOne.letter, letterOne.bestCoords[1], letterOne.bestCoords[0], letterOne.bestScore))
                matchLog.write('%s|%s|%d|%d|%f|win\n' % (self.cellLabel, letterTwo.letter, letterTwo.bestCoords[1], letterTwo.bestCoords[0], letterTwo.bestScore))
                searchOrder[ii] = letterTwo
                searchOrder[endPoint] = letterOne
      # If somehow no matches were found, treat the content same as a black blocked square, and assign it
      # same empty set symbol as used in that scenario.
      matchWin = searchOrder[0]
      self.contentString = matchWin.letter + matchWin.tiedWith
      if self.contentString == '':
         self.contentString = '-'
      return self.contentString


class CrosswordTicket:
   ticketSIFT = cv2.xfeatures2d.SIFT_create(TICKET_MAX_FEATURES, TICKET_SIFT_OCTAVES, 0, 0, TICKET_SIFT_SIGMA)
   
   EXEMPLARS = { }
   ALPHABET = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z"
   ALPHABET_SPLIT = ALPHABET.split(' ')
   # eSIFT = cv2.xfeatures2d.SIFT_create(CELL_MAX_FEATURES, CELL_SIFT_OCTAVES, 0, 0, CELL_SIFT_SIGMA)
   for letter in ALPHABET_SPLIT:
      EXEMPLARS[letter] = LetterMatcher(letter) #, eSIFT)

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
      self.ticketImg = SiftImage(ticketFilePath, ticketImg, CrosswordTicket.ticketSIFT)
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
      # print extractedImg
      # print sampleDir + '/warpGridCrop.tif'
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
      # Translate a border around the crossword grid in the query image into the coordinate system of the
      # ticket being analyzed, using keyfeature correlation to identity the kernel matrix required to do so.
      foundInRegion = np.int32(cv2.perspectiveTransform(np.float32(CrosswordTicket.gridLocation), M))
      decoratedImg = ticketImg.copy()
      decoratedImg = cv2.drawKeypoints(decoratedImg, kpT, None, color=(0,128,192)) #, flags=0)
      decoratedImg = cv2.polylines(decoratedImg, [foundInRegion], True, 160, 5, cv2.LINE_AA)
      # decoratedImg = cv2.polylines(decoratedImg, [foundInRegion], True, (96,0,224), 5, cv2.LINE_AA)
      cv2.imwrite(sampleDir + '/warpFullTicket.tif', decoratedImg)
      # Create side-by-side visualization of the perspective transformation's source match support.
      draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                         singlePointColor = (224,0,96), # draw unmatched keyponts in red?
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
            cellCoordsStr = '%dx%d' % (ii, jj)
            cellFilePath = cellFileTemplate % (ii,jj)
            matchFileTemplate = sampleDataDir + '/match-%s_' + cellCoordsStr + '_%s.tif'
            cellObject = TicketCell(cellImg, cellLabel, cellFilePath,  matchFileTemplate)
            xwGrid[ii][jj] = cellObject.identifyContent(CrosswordTicket.EXEMPLARS, matchLog)
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
queryObj = SiftImage('ticketQuery', queryImg, CrosswordTicket.ticketSIFT)

# Iterate over the training images
for ii in range(1,19):
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

