# ICE-LAPSE

This repository contains the sourse code used for analysing the image dataset acquired within the project ICE-LAPSE "Analysis of Antarctic benthos dynamics by using non-destructive monitoring devices and permanent stations"  (PNRA 2013/AZ1.16) of the Italian Amtarctic Program.

Three tools are provided with the aim of estimating:
- the underwater environmental diffused light;
- the organisms activity dynamics;
- the sponge activity;

This document briefly describes how to execute the proposed analysis software tools. 
All the proposed software tools are written in Python 3 and need the following Python packages:

- sys
- os 
- numpy 
- datetime
- multiprocessing
- functools
- PIL
- CV2

Three scripts have been developed as reported below together with the corresponding input and output arguments:

Scritpt Name              |  Input Arguments  |  Output
estimateDiffusedLight.py  |  imgTimeSeries    |  resTimeSeries
estimateActivity.py       |  imgTimeSeries    |  mask resTimeSeries
estimateOsculumRadius.py  |  imgTimeSeries    |  resTimeSeries


imgTimeSeries: is a text file where each row represents an analyzed image and contains the corresponding acquisition time (year, month, day, hour, minute, second) and the full path of the image. In the case of the tool estimateOsculumRadius.py, the input image time series contains the oscua's cropped images. The image time series are formatted according to the following schema:

YYYY-MM-DD hh:mm:ss imgPath
.
.
.
YYYY-MM-DD hh:mm:ss imgPath

mask: is a binary image with the same size of the analyzed images

resTimeSeries: is a text file where each row contains the acquisition time (year, month, day, hour, minute, second) of the image and the analysis result. Each row is formatted according to the following schema:

YYYY-MM-DD hh:mm:ss value
.
.
.
YYYY-MM-DD hh:mm:ss value
