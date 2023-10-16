#---------------------------ooo0ooo---------------------------
#       Aurora Monitor v5.0
#       updated 26_12_18
#       now using threading
#       Ian Robinson
#       starfishprime.co.uk
#
#---------------------------Notes---------------------------
#
#
#---------------------------ooo0ooo---------------------------

import smbus
import time
import os
import gc




from threading import Thread
import numpy as np
from obspy import UTCDateTime, read, Trace, Stream
from obspy.signal.filter import bandpass, lowpass, highpass
from shutil import copyfile
import copy


import matplotlib
matplotlib.use('Agg') #prevent use of Xwindows

#---------------------------ooo0ooo---------------------------
#---------------------------ooo0ooo---------------------------

#---------------------------ooo0ooo---------------------------
def readPressure():
    rawdata = bytearray
    z=bytearray
    rawPressure = int
    rawdata=[]

    z = [0,0,0,0]
    z = sensor.read_i2c_block_data(addr, 0, 4) #offset 0, 4 bytes

    dMasked = (z[0] & 0x3F)
    rawdata = (dMasked<<8) +  z[1]    

    rawpressure = int(rawdata)
    pressure = (((rawpressure-8192.0)/16384.0)*497.68)
    
    return pressure

#---------------------------ooo0ooo---------------------------
def SaveDataMSeed(st, StartDateTime, nSamples, stationid):


  Year = str(StartDateTime.year)
  Month = str(StartDateTime.month)
  Day = str(StartDateTime.day)
  Hour = str(StartDateTime.hour)
  Minute = str(StartDateTime.minute)

  yearDir =  'Data' + '/' + Year

  FileName = str(stationid) + '_'+ str(Year) + '_' + str(Month) + '_' + str(Day) + '__' + Hour +':' + Minute +'.mseed'
  
  here = os.path.dirname(os.path.realpath(__file__))

  try:
    os.makedirs(yearDir)
  except OSError:
    if not os.path.isdir(yearDir):
        raise

  FilePath = os.path.join(here, yearDir, FileName)
  
  dataFile = open(FilePath, 'wb')
  st.write(dataFile,format='MSEED',  encoding=4, reclen=256)
  dataFile.close()
  print ('Data write of active data completed')

###---------------------------ooo0ooo---------------------------
def PlotDayPlot(st, StartDateTime, nSamples, lowCut, highCut):
    print('plotting DayPlot - nSamples=', nSamples)

    if (nSamples > 3000):
        try:

            st.filter('bandpass', freqmin=lowCut, freqmax=highCut, corners=4, zerophase=True)


            Year = str(StartDateTime.year)
            Month = str(StartDateTime.month)
            Day = str(StartDateTime.day)


            saveDir =  'Plots/' 
                
            dateString = str(Year) + '_' + str(Month) + '_' + str(Day)
            plotTitle= str(lowCut) + '-' + str(highCut) + ' Hz --' + dateString 
            filename1 = ('Plots/Today.svg')
            filename2 = 'Plots/' + dateString + '.svg'

            
            st.plot(type="dayplot",outfile=filename1, title=plotTitle, data_unit='$\Delta$Pa', interval=60, right_vertical_labels=False, one_tick_per_line=False, color=['k', 'r', 'b', 'g'], show_y_UTC_label=False)
            #old st[0].
            
            copyfile('Plots/Today.svg', filename2)

        except (ValueError,IndexError):
            print('an index error on plotting dayplot!')

###---------------------------ooo0ooo---------------------------
def createMSeed(DataArray, StartDateTime, nSamples, stationid):

    EndDateTime = UTCDateTime()
    ActualSampleFrequency = float(nSamples) / (EndDateTime - StartDateTime)

    # Fill header attributes
    stats = {'network': 'IR', 'station': stationid, 'location': ' ',
         'channel': '1', 'npts': nSamples, 'sampling_rate': ActualSampleFrequency,
         'mseed': {'dataquality': 'D'}}
    # set current time
    stats['starttime'] = StartDateTime
    st = Stream([Trace(data=DataArray[0:nSamples], header=stats)])
    return st

###---------------------------ooo0ooo---------------------------
def PlotAcousticPower(st, StartDateTime, nSamples, lowCut, highCut, deltaT):

    print('plotting Acoustic Power DayPlot')

    if (nSamples > 3000):
        try:
            tr = CalcRunningMeanPower(st, deltaT, lowCut, highCut)
           
            Year = str(StartDateTime.year)
            Month = str(StartDateTime.month)
            Day = str(StartDateTime.day)

            saveDir =  'Plots/' 
                
            dateString = str(Year) + '_' + str(Month) + '_' + str(Day)
            plotTitle= 'AcousticPower ' + str(lowCut) + '-' + str(highCut) + ' Hz --' + dateString
            filename1 = ('Plots/TodayAcousticPower.svg')
            filename2 = 'Plots/' + dateString + 'AcousticPower.svg'

            #tr.plot(type="dayplot", title=plotTitle, data_unit='W', outfile=filename1, interval=60, right_vertical_labels=False, one_tick_per_line=True, color=['k', 'r', 'b', 'g'], show_y_UTC_label=True)
            tr.plot(type="dayplot",outfile=filename1, title=plotTitle, data_unit='$Wm^{-2}$', interval=60, right_vertical_labels=False, one_tick_per_line=False, color=['k', 'r', 'b', 'g'], show_y_UTC_label=False)
                
            copyfile('Plots/TodayAcousticPower.svg', filename2)
            print('Acoustic Power Plotted')

        except (ValueError,IndexError):
            print('an index error on plotting acoustic power!')
 
###---------------------------ooo0ooo---------------------------
def CalcRunningMeanPower(st, deltaT, lowCut, highCut):

    N=len(st[0].data)
    dt = st[0].stats.delta
    newStream=st[0].copy()

    newStream.filter('bandpass', freqmin=lowCut, freqmax=highCut, corners=4, zerophase=True)
    x=newStream.data
   
    x=x**2
    
    nSamplePoints=int(deltaT/dt)
    runningMean=np.zeros((N-nSamplePoints), np.float32)

    #determine first tranche
    tempSum = 0.0
    
    for i in range(0,(nSamplePoints-1)):
        tempSum = tempSum + x[i]

        runningMean[i]=tempSum

    #calc rest of the sums by subracting first value and adding new one from far end  
    for i in range(1,(N-(nSamplePoints+1))):
            tempSum = tempSum - x[i-1] + x[i + nSamplePoints]
            runningMean[i]=tempSum
    # calc averaged acoustic intensity as P^2/(density*c)
    runningMean=runningMean/(1.2*330)


    newStream.data=runningMean
    newStream.stats.npts=len(runningMean)

    return newStream

###---------------------------ooo0ooo---------------------------
def SaveAndPlot(DataArray, StartDateTime, nSamples, stationid,):

    st = createMSeed(DataArray, StartDateTime, nSamples, stationid)

    SaveDataMSeed(st, StartDateTime, nSamples, stationid)
    #PlotAcousticPower(st, StartDateTime, nSamples, lowCut, highCut, deltaT)
    #PlotDayPlot(st, StartDateTime, nSamples, lowCut, highCut)






###--------------------------Main Body--------------------------
###-                                                           -
###-                                                           -
###--------------------------+++++++++--------------------------

sensor = smbus.SMBus(1)
addr = 0x28
os.chdir('/home/pi/InfraSound')


stationid = 'Beta'
SamplingFrequencyHz=40.00


SamplingPeriodInSeconds = 1.00/SamplingFrequencyHz

DataFileLengthSeconds=3600*24

graphingInterval = 300 # time between saves in seconds

deltaT = 5.0  # time interval seconds to calculate running mean for acoustic power

#-------------create folders if absent------------------------
try:
    os.makedirs('Plots/')
except OSError:
    if not os.path.isdir('Plots/'):
        raise
###---------------------------ooo0ooo---------------------------
# -- create numpy array to store pressure and time data
TargetNoSamples = int(DataFileLengthSeconds*SamplingFrequencyHz*1.1)
DataArray=np.zeros(TargetNoSamples, np.float32) 
nSamples = 0

lowCut=0.01         # low cut-off frequency
highCut=10.0        # low cut-off frequency


StartDateTime=UTCDateTime() #initialise variable
EndDateTime=UTCDateTime() #initialise variable
lastSaveTime=UTCDateTime() #initialise variable



while 1:
        
    time.sleep(SamplingPeriodInSeconds)
    
    try:
        DataArray[nSamples] = readPressure()
        nSamples = nSamples + 1
    except IOError:
        print('read error')
        DataArray[nSamples] = 0.00
        nSamples = nSamples + 1


    if (StartDateTime.day != UTCDateTime().day):
        
        threadSaveAndPlot = Thread(target=SaveAndPlot, args=(DataArray, StartDateTime, nSamples, stationid,))
        threadSaveAndPlot.start()

        DataArray=np.zeros(TargetNoSamples, np.float32)  #reset data array to zero for new day
        StartDateTime=UTCDateTime()
        lastSaveTime=UTCDateTime()
        nSamples = 0
        gc.collect

    if ((UTCDateTime() - lastSaveTime) > graphingInterval):

        threadSaveAndPlot = Thread(target=SaveAndPlot, args=(DataArray, StartDateTime, nSamples, stationid,))
        threadSaveAndPlot.start()

        lastSaveTime = UTCDateTime()


