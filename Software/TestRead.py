import smbus
import time
import binascii

import serial
import numpy as np
from datetime import datetime
from datetime import timedelta
import time
import ftplib
import string
import matplotlib
matplotlib.use('Agg') #prevent use of Xwindows
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.dates as mdates
import statistics   #used by median filter
import os
import gc

 #---------------------------ooo0ooo---------------------------

sensor = smbus.SMBus(1)
addr = 0x28






 #---------------------------ooo0ooo---------------------------
def bytes_to_int(bytes):
  return int(bytes.encode('hex'), 16)

 #---------------------------ooo0ooo---------------------------
def readPressure():
    rawdata = bytearray
    z=bytearray
    rawPressure = int
    rawdata=[]
    #z= sensor.read_word_data(addr,2)
    #print z
    z = [0,0,0,0]
    z = sensor.read_i2c_block_data(addr, 0, 4) #offset 0, 4 bytes

    dMasked = (z[0] & 0x3F)
    rawdata = (dMasked<<8) +  z[1]    

    rawpressure = int(rawdata)
    pressure = (((rawpressure-8192.0)/16384.0)*497.68)
    
    return pressure

 #---------------------------ooo0ooo---------------------------

 #---------------------------main--------------------------

os.chdir('/home/pi/InfraSound')

Attempts=0
Fails=0
Pressure=-999.99

while (Attempts < 1000):
  time.sleep(0.01)
  Attempts=Attempts+1

  try:
    Pressure=readPressure()
  except IOError:
    Fails=Fails+1
    #print('read fail')

      


print ('pressure - ', Pressure, '    attempts - ', Attempts, '   failed - ', Fails)
        


    


