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
def plotTodaysGraph(ActiveData):


    if len(ActiveData[0])>10:
        yearNow=datetime.now().year
        monthNow=datetime.now().month
        dayNow=datetime.now().day

        
        filename=('Today.png')
        filename2=str(datetime.now().strftime('%d'+"_"+'%b'+"_"+'%Y'))+'.png'
        
        majorLocator   = MultipleLocator(3)
        majorFormatter = FormatStrFormatter('%d')
        minorLocator   = MultipleLocator(1)
            


        existingTodaysData=False
        
        xdata=[]
        ydata=[]
        YAxisRange = 2.0
        FirstPointToday=True
        
        zeroPoint=ActiveData[1][0] 
        
        
        for i in range (len(ActiveData[0])-1):
            if ActiveData[0][i].year == yearNow:
                if ActiveData[0][i].month == monthNow:
                    if ActiveData[0][i].day == dayNow:
                        existingTodaysData=True
                        if(FirstPointToday==True):
                            zeroPoint=ActiveData[1][i]
                            FirstPointToday=False

                        time=ActiveData[0][i].hour +(ActiveData[0][i].minute/60) +(ActiveData[0][i].second/3600)                        
                        xdata.append(time)
                        B=ActiveData[1][i]-zeroPoint
                        ydata.append(B)
                        if(abs(B)>YAxisRange):
                            YAxisRange=abs(B*1.2) #rescale axis



        if(len(ydata) > 6):
            print('todays data exists ', len(ydata),' data points')


            dateHeader=str(datetime.now().strftime('%d'+" "+'%b'+" "+'%Y'))

            print ('Today', dateHeader)
            upDated=str(datetime.now().strftime('%H')+":"+datetime.now().strftime('%M'))
                
            
            fig = plt.figure()
                
            xAxisText="Time  updated - "+ upDated +" UTC"
            axes = plt.gca()
            axes.set_xlim([0,24])
            axes.xaxis.set_major_locator(majorLocator)
            axes.xaxis.set_major_formatter(majorFormatter)
            axes.xaxis.set_minor_locator(minorLocator)

                
            plt.title(dateHeader)
            plt.xlabel(xAxisText)

            zeroLabel=str('%0.5g' % (zeroPoint/1000))
            plt.ylabel('$\Delta$Flux Density - nT     0.0='+zeroLabel+'$\mu T$')
            axes.relim()
            plt.ylim(((-YAxisRange),(+YAxisRange)))
            plt.plot(xdata, ydata,  marker='None',    color = 'darkolivegreen')

            plt.grid(True)

            plt.savefig(filename)
            plt.savefig(filename2)
                
            print('Plotting Todays Graph -zeroP=', zeroPoint)

            plt.close(fig)
            plt.clf()  

        del xdata, ydata
        
    return
    #---------------------------ooo0ooo---------------------------
  #scans for any data points older than 24 hrs and appends them to an archive file

def ArchiveOldData():
    global ActiveData

    keepInterval= timedelta(hours=24)
    print('Archiving Old data')
    
    newData=[[],[],[]]
    archiveData=[[],[],[]]

    if len(ActiveData[0])>5:
        for i in range (len(ActiveData[0])-1):
            if (datetime.now()-(ActiveData[0][i]))>keepInterval:
                archiveData[0].append(ActiveData[0][i])
                archiveData[1].append(ActiveData[1][i])
            else:
                newData[0].append(ActiveData[0][i])
                newData[1].append(ActiveData[1][i])

        if len(archiveData[0])>0:
            dataFile = open('ArchivedData.dat', 'a')
            for i in range (len(archiveData[0])-1):
                ##open file/save close file
                dateString=archiveData[0][i].strftime(DB_TIME_FORMAT)
                fieldString=str(archiveData[1][i])
                fieldString=fieldString[:10] #truncate string

                
            dataFile.close()
        ActiveData=[[],[]]
        ActiveData=newData
            
    print('Archiving Old data--', len(archiveData[0]),' datapoints archived')
    WriteActiveDataFile(ActiveData)

    return
#---------------------------ooo0ooo---------------------------
def WriteActiveDataFile(ActiveData):

    dataFile = open('ActiveData.dat', 'w')

    for i in range(len(ActiveData[0])):
        dateString=ActiveData[0][i].strftime(DB_TIME_FORMAT)
        fieldString=str(ActiveData[1][i])
        fieldString=fieldString[:10] #truncate string

        dataFile.write(dateString + " , " + fieldString + " , "  + "\n")
   
    dataFile.close()
    print ('Data write of active data completed')

    return
#---------------------------ooo0ooo---------------------------
 #---------------------------ooo0ooo---------------------------
 #---------------------------ooo0ooo---------------------------

os.chdir('/home/pi/InfraSound')

while 1:
  time.sleep(0.1)
  pressure=readPressure()
  print ('pressure - ', pressure)
        


    


