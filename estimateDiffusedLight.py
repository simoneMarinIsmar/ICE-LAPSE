#! /usr/bin/python3

import sys
import numpy as np
import datetime as dt
import multiprocessing as mp
from PIL import Image

##########################################################

# https://github.com/joyeecheung/dark-channel-prior-dehazing
#R, G, B = 0, 1, 2  # index for convenience
#L = 256  # color depth


def getDarkChannel(I, w):
    """Get the dark channel prior in the (RGB) image data.
    Parameters
    -----------
    I:  an M * N * 3 numpy array containing data ([0, L-1]) in the image where
        M is the height, N is the width, 3 represents R/G/B channels.
    w:  window size
    Return
    -----------
    An M * N array for the dark channel prior ([0, L-1]).
    """
    M, N, _ = I.shape
    padded = np.pad(I, ((int(w / 2), int(w / 2)), (int(w / 2), int(w / 2)), (0, 0)), 'edge')
    darkCh = np.zeros((M, N))
    brightCh = np.zeros((M, N))
    diffCh = np.zeros((M, N))
    correctedDarkCh = np.zeros((M, N))
    for i, j in list(np.ndindex(darkCh.shape)):
        darkCh[i, j] = np.min(padded[i:i + w, j:j + w, 1:3])  # CVPR09, eq.5
        brightCh[i, j] = np.max(padded[i:i + w, j:j + w, 1:3])
        diffCh[i, j] = abs(brightCh[i, j] - darkCh[i, j])
        correctedDarkCh[i, j] = max(0, (darkCh[i,j] - .5*diffCh[i,j]))

    return darkCh, brightCh, diffCh, correctedDarkCh


def getDiffusedLight(I, darkch, p):
	
	# reference CVPR09, 4.4
	M, N = darkch.shape
	flatI = I.reshape(M * N, 3)
	flatdark = darkch.ravel()
	searchidx = (-flatdark).argsort()[:int(M * N * p)] #find top M * N * p indexes
	fIndices = list(filter(lambda x: flatdark[x] != 0, searchidx))

	if any(fIndices):
		#different coordinate system between Pilow imgages and numpy array
		return max(np.mean(flatI.take(fIndices, axis=0), axis=0))
	else:
		return 0


def getDateFromStr(strDateTime):
	# 2017-02-27 05:16:13

	strDate, strTime = strDateTime.split(' ')
	YY, MM, DD = strDate.split('-')
	HH, mm, ss = strTime.split(':')
	
	return dt.datetime(int(YY), int(MM), int(DD),\
		int(HH), int(mm), int(ss))


def readTimeSeries(fileName):
	dataSet = {}
	inFile = open(fileName, 'r')

	rows = []
	for row in inFile:
		strDate, name = row.split('\t')
		dataSet[getDateFromStr(strDate)] = name.strip()
	inFile.close()
	
	return dataSet


def processImage(timeSeriesElem):
    print(timeSeriesElem[1], flush=True)
    im = Image.open(timeSeriesElem[1])
    I = np.asarray(im, dtype=np.float64)
    dcpArray, bcpArray, diffArray, cdcArray = getDarkChannel(I, 101)
    diffusedLight = getDiffusedLight(I, cdcArray, .3)

    return (timeSeriesElem[0], diffusedLight)

	
def writeTimeSeries(timeSeries):
	outFile = open('diffusedLight.csv', 'w')
	
	for dateTime, value in timeSeries:
		outFile.write(str(dateTime)+'\t'+str(value)+'\n')

	outFile.close()


def usage():
	print('./estimateDiffusedLight.py timeSeriesInputFileName')


###############################################################

if __name__ == '__main__':

    if len(sys.argv) != 2:
        usage()
        sys.exit(1)

    timeSeriesFileName = sys.argv[1]
    #diffusedLightTimeSeriesFileName = sys.argv[2]

    imgDS = readTimeSeries(timeSeriesFileName)

    timeSeries = list(imgDS.items()) # [...(dateTime, imgName)...]
    timeSeries.sort(key=lambda x:x[0])

    diffusedLightTimeSeries = []
    for elem in timeSeries:
        diffusedLightTimeSeries.append(processImage(elem))

    # multiprocess image processing
	#pool = mp.Pool(mp.cpu_count())
	#diffusedLightTimeSeries = pool.map(processImage, timeSeries)
	#pool.close()
	#pool.join()

    diffusedLightTimeSeries.sort(key=lambda x:x[0])
    normalisedDiffusedLightTimeSeries = map(lambda x: (x[0], float(x[1])/255), diffusedLightTimeSeries)

    writeTimeSeries(normalisedDiffusedLightTimeSeries)
