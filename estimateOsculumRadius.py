#! /usr/bin/python3

import sys
import os
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


def getCroppedImage(imgFileName, mask=None):
	
	if not os.path.exists(imgFileName):
		return np.array([])
	
	img = cv2.imread(imgFileName).astype(np.uint8)
	
	if mask != None:
		imGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) & mask
		
		cropIndices = np.nonzero(imGray)
		croppedImg = imGray[min(cropIndices[0]):max(cropIndices[0])+1,\
			min(cropIndices[1]):max(cropIndices[1])+1]
		return croppedImg
	
	else:
		return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	#croppedImg = cv2.medianBlur(croppedImg,5)
	#return croppedImg


def getEntropy(img):
	#hist = np.array(getMaskedHistogram(img, cnt))
	hist = cv2.calcHist([img],[0],None,[256],[0,256])
	sumHist = float(sum(hist))
	normHist = hist/sumHist
	
	ent = -sum(map(lambda x: x*np.log(x)/np.log(2), filter(lambda x: x!=0, normHist)))[0]
	
	return ent


def getCircleRadius(name, mask=None):
	print('.'),
	sys.stdout.flush()
	
	croppedImg = getCroppedImage(name, mask)
	if not croppedImg.any():
		return None
	
	cropedImgEntropy = getEntropy(croppedImg)
	
	#if cropedImgEntropy > 6.6: # Osculum 3
	if cropedImgEntropy > 0.0: # Osculum 1
	
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
		eImg = clahe.apply(croppedImg)
		bImg = cv2.blur(eImg,(7,7))
	
		#numRows, numCols = croppedImg.shape
		#GOOD circles = cv2.HoughCircles(bImg,cv2.HOUGH_GRADIENT,2,1,\
			#param1=80,param2=40,minRadius=15,maxRadius=50)
			#OSCULUM1, OSCULUM3 = param2=40, minRadius=15
			#da verificare OSCULUM2 = param2=50, minRadius=10
			#OSCULUM2: param1=80,param2=30,minRadius=10,maxRadius=35
			#OSCULUM4 = param2=35, minRadius=10
		circles = cv2.HoughCircles(bImg,cv2.HOUGH_GRADIENT,2,1,\
			param1=80,param2=30,minRadius=10,maxRadius=35)
	
		if circles is not None:
			circles = np.uint16(np.around(circles))
			avgC = np.average(circles[0],axis=0).astype(int)
			return (name, avgC)
		else:
			return (name, (0,0,0))
	else:
		return (name, (0,0,0))
	

def writeCircleRadiusData(circleRadiusData, imgDatesTS):
	outFile = open('./osculumRadiusData.csv', 'w')
	
	data = [(imgDatesTS[d[0]], d[1][2]) for d in circleRadiusData if d != None]
	data.sort(key=lambda x:x[0])
	
	if len(data):
		print('\n', len(data), 'elements to save')
		for d in data:
			outFile.write(str(d[0])+'\t'+str(d[1])+'\n')
	else:
		print('\nNo circles found\n')
	
	outFile.close()
	print('./osculumRadiusData.csv saved')
	

def usage():
	print('./estimateOsculumRadius.py imageTimeSeriesFileName\n')
	
	

###############################################################


if __name__ == '__main__':

    if len(sys.argv) != 2:
        usage()
        sys.exit(1)

    # read parameters
    imageTimeSeriesFileName = sys.argv[1]

    # read time series
    imgNamesTS, imgDatesTS = readImageTimeSeriesFile(imageTimeSeriesFileName)
    dates, names = zip(*sorted(imgNamesTS.items()))


    #if len(sys.argv) == 4:
    pGetCircleRadius = ft.partial(getCircleRadius, mask=None)
    pool = mp.Pool(mp.cpu_count())
    circleRadiusData = pool.map(pGetCircleRadius, names)
    pool.close()
    pool.join()
    
    writeCircleRadiusData(circleRadiusData, imgDatesTS)
