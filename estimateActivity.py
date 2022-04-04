#! /usr/bin/python3

import sys
import numpy as np
import datetime as dt
import multiprocessing as mp
import functools as ft
import cv2


###############################################################

def getDateFromStr(strDateTime):
	# 2017-02-27 05:16:13

	strDate, strTime = strDateTime.split(' ')
	YY, MM, DD = strDate.split('-')
	HH, mm, ss = strTime.split(':')
	
	return dt.datetime(int(YY), int(MM), int(DD),\
		int(HH), int(mm), int(ss))
	

def readImageTimeSeriesFile(imageSeq):
	# read input sequence
	imgNamesTS = {}
	imgDatesTS = {}
	inFile = open(imageSeq, 'r')
	for row in inFile:
		strDateTime, name = row.split('\t')
		name = name.strip()
		date = getDateFromStr(strDateTime)
		imgNamesTS[date] = name
		imgDatesTS[name] = date
	inFile.close()
	
	return imgNamesTS, imgDatesTS


def readImgTriplet(names, index):
	triplet = []
	if index > 0 and (index < len(names)-1):
		for i in range(-1, 2):
			#print names[index+i]
			im = cv2.imread(names[index+i])
			triplet.append(im)
			
	return triplet


def selectContours(contours, hierarchy, minContLen):
	selectedCnt = []

	for i,cnt in enumerate(contours):
		if (hierarchy[0][i][3] == -1) and (len(cnt) > minContLen):
			selectedCnt.append(cnt)

	return selectedCnt


def morphologyClosure(img, kDSize, dIteration, kESize, eIterations):

	kernelDilation = np.ones((kDSize,kDSize), np.uint8)
	dilatedImg = cv2.dilate(img, kernelDilation, iterations = dIteration)

	kernelErosion = np.ones((kESize,kESize), np.uint8)
	erodedImg = cv2.erode(dilatedImg, kernelErosion, iterations = eIterations)

	return erodedImg


def pairDiff(img1, img2):

	diff = cv2.absdiff(img1, img2)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,2))
	eDiff = clahe.apply(diff)
	#eDiff = cv2.equalizeHist(diff)

	thDiff = cv2.adaptiveThreshold(eDiff,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,-3)
	
	kernelErosion = np.ones((3,3), np.uint8)
	erDiff = cv2.erode(thDiff, kernelErosion, iterations = 1)
	mDiff = morphologyClosure(erDiff, 7, 2, 5, 1)
	
	#return eDiff, mDiff
	return mDiff


def imgDiff(triplet, mask):
	
	# masking
	mTriplet = map(lambda x: x&mask, triplet)
	
	# blurring
	bTriplet = list(map(lambda x: cv2.GaussianBlur(x,(3,3),0), mTriplet))
	
	# differencing
	diff = cv2.absdiff(bTriplet[1], bTriplet[0]) & cv2.absdiff(bTriplet[1], bTriplet[2])
	#cv2.imshow('diff', diff)
	
	# thresholding
	thDiff = cv2.adaptiveThreshold(diff,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,9,-5)
	#cv2.imshow('thDiff', thDiff)
	
	# morphology operations
	kernelErosion = np.ones((3,3), np.uint8)
	erDiff = cv2.erode(thDiff, kernelErosion, iterations = 1)
	#cv2.imshow('erDiff', erDiff)
	mDiff = morphologyClosure(erDiff, 9, 3, 3, 1)
	#cv2.imshow('mDiff', mDiff)
	
	#contours extraction
	contours, hierarchy = cv2.findContours(mDiff,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	selectedCnts = selectContours(contours, hierarchy, 1)
	
	return selectedCnts, diff


def imageDifferencing(imgIndex, imgNames, imgMask):
	triplet = readImgTriplet(imgNames, imgIndex)
	img = triplet[1]
	gTriplet = map(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), triplet)
	cnts, diff = imgDiff(gTriplet, imgMask)
	
	return (imgNames[imgIndex], cnts)


def extractBackground(imgNames, mask, shape):
	acc = np.float32(np.zeros(shape))
	#for name in imgNames[236:542]:
	for name in imgNames:
		img = cv2.cvtColor(cv2.imread(name),cv2.COLOR_BGR2GRAY)
		if img != []:
			cv2.accumulateWeighted(img, acc, .05, mask)

	return cv2.convertScaleAbs(acc)


def extractRangeBackground(imgNames, mask, index, radius, shape):
	acc = np.float32(np.zeros(shape))
	if index < radius:
		start = 0
		stop = radius+1
	else:
		start = index-radius
		stop = index+1
	for name in imgNames[start:stop]:
		img = cv2.cvtColor(cv2.imread(name),cv2.COLOR_BGR2GRAY)
		if img != []:
			cv2.accumulateWeighted(img, acc, .05, mask)

	clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
	eAcc = clahe.apply(cv2.convertScaleAbs(acc))
	#eAcc = cv2.equalizeHist(cv2.convertScaleAbs(acc))

	#return cv2.convertScaleAbs(acc)
	return eAcc


def bckgroundSub(bckg, img):
	
	diff = pairDiff(bckg, img)
	#cv2.imshow('diff', diff)
	
	im, contours, hierarchy = cv2.findContours(diff,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	selectedCnts = selectContours(contours, hierarchy, 5)
	
	return selectedCnts, diff


def backgroundSubtraction(names, imgIndex, mask):
	img = cv2.imread(names[imgIndex])
	imGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#extract background
	#bckg = extractBackground(names, mask, mask.shape)
	bckg = extractRangeBackground(names, mask, imgIndex, 15, mask.shape)
	cv2.imshow('background', bckg)
	cv2.waitKey(-1)
		
	# make background subtraction
	cnts, diff = bckgroundSub(bckg&mask, imGray&mask)
	
	return img, cnts, diff


def writeImgData(imgDatesTS, imgData):
	
	outFile = open('./activity.csv', 'w')
	for data in imgData:
		outFile.write(str(imgDatesTS[data[0]])+'\t'+str(len(data[1]))+'\n')
	outFile.close()
	

def usage():
	print('\n./estimateActivity.py imageTimeSeriesFileName imgMaskFileName')

###############################################################


if __name__ == '__main__':

    if len(sys.argv) == 3:
	    # read parameters
        imageTimeSeriesFileName = sys.argv[1]
        imgMaskFileName = sys.argv[2]

        # read time series
        imgNamesTS, imgDatesTS = readImageTimeSeriesFile(imageTimeSeriesFileName)
        dates, names = zip(*sorted(imgNamesTS.items()))

        # read mask
        mask = cv2.cvtColor(cv2.imread(imgMaskFileName),cv2.COLOR_BGR2GRAY)

        indices = range(1, len(names)-1)
        pImageDifferncing = ft.partial(imageDifferencing, imgNames=names, imgMask=mask)
        pool = mp.Pool(mp.cpu_count())
        imgData = pool.map(pImageDifferncing, indices)
        pool.close()
        pool.join()
		
        writeImgData(imgDatesTS, imgData)
	
    else:
        usage()
        sys.exit(1)
