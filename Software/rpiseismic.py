#---------------------------ooo0ooo---------------------------
#       Seismic Monitoring Software
#       updated Sept 2020
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
	#this function specifically reads data over i2c from a 
	#DLVRF50D1NDCNI3F Amphenol mems differential pressure sensor 
	
	# this routine will need to be changed for a diffent input device
	#such as a voltage via an a/d convertor
	# note: the return value 'pressure' is a floating point number
	
    rawdata = bytearray
    z=bytearray
    rawPressure = int
    rawdata=[]

    z = [0,0,0,0]
    z = sensor.read_i2c_block_data(addr, 0, 4) #offset 0, 4 bytes

    dMasked = (z[0] & 0x3F)
    rawdata = (dMasked<<8) +  z[1]    

    rawpressure = int(rawdata)
    pressure = (((rawpressure-8192.0)/16384.0)*250.0 *1.25)
    
    return pressure

#---------------------------ooo0ooo---------------------------
def SaveDataMSeed(st, HourlyStartTime):
	Year = str(HourlyStartTime.year)
	Month = str(HourlyStartTime.month)
	Day = str(HourlyStartTime.day)
	Hour = str(HourlyStartTime.hour)
	
	saveDir =  'Data' + '/' + Year + '/' + Month+ '/' + Day
	FileName = Hour + '.mseed'
  
#----------create data Directory Structure if not already present  
	here = os.path.dirname(os.path.realpath(__file__))
	

	try:
		os.makedirs('Data/'+Year+'/')
	except OSError:
		if not os.path.isdir('Data/'+Year+'/'):
			raise
 
	try:
		os.makedirs('Data/'+Year+'/'+Month+'/')
	except OSError:
		if not os.path.isdir('Data/'+Year+'/'+Month+'/'):
			raise

	try:
		os.makedirs('Data/'+Year+'/'+Month+'/'+Day+'/')
	except OSError:
		if not os.path.isdir('Data/'+Year+'/'+Month+'/'+Day+'/'):
			raise


	FilePath = os.path.join(here, saveDir, FileName)
  
	dataFile = open(FilePath, 'wb')
	st.write(dataFile,format='MSEED', encoding=4, reclen=4096)
	dataFile.close()
	print ('Data write of active data completed')

###---------------------------ooo0ooo---------------------------
def PlotDayPlot(st, DailyStartTime, nDailySamples, lowCut, highCut, stationid, stationchannel, location):
	#produces a 24hour obspy 'dayplot' saved in svg format.
	#two copies are produced Today.svg and 'date'.svg

	here = os.path.dirname(os.path.realpath(__file__))

	
	Year = str(DailyStartTime.year)
	Month = str(DailyStartTime.month)
	Day = str(DailyStartTime.day)
	yearDir =  'Data' + '/' + Year
	
	StationInfo = str(stationid)+'-'+str(stationchannel)+'-'+str(location)
	
	if (nDailySamples > 3000):
		try:
			saveDir =  'Plots/' 
			dateString = str(Year) + '_' + str(Month) + '_' + str(Day)
			plotTitle= 'Raw Pressure ' +':::'+StationInfo+':::'+' '+dateString+'  '+str(lowCut) + '-' + str(highCut) + ' Hz'
			filename1 = ('Plots/Today.svg')
			filename2 = 'Plots/' + dateString+'__'+StationInfo+'__RawPressure.svg'
			
			st.plot(type="dayplot",outfile=filename1, title=plotTitle, data_unit='$\Delta$Pa', interval=60, right_vertical_labels=False, one_tick_per_line=False, color=['k', 'r', 'b', 'g'], show_y_UTC_label=False)
			copyfile('Plots/Today.svg', filename2)
			print ('Plotting of active data completed')
		except (ValueError,IndexError):
			print('an index error on plotting dayplot!')

###---------------------------ooo0ooo---------------------------
def createMSeed(Readings, StartTime, EndTime, nSamples, stationid, stationchannel, location):

    ActualSampleFrequency = float(nSamples) / (EndTime - StartTime)

    # Fill header attributes
    stats = {'network': 'IR', 'station': stationid, 'location': location,
         'channel': stationchannel, 'npts': nSamples, 'sampling_rate': ActualSampleFrequency,
         'mseed': {'dataquality': 'D'}}
    # set current time
    stats['starttime'] = StartTime
    st = Stream([Trace(data=Readings[0:nSamples], header=stats)])
    return st

###---------------------------ooo0ooo---------------------------
def PlotAcousticPower(st, DailyStartTime, nDailySamples, lowCut, highCut, deltaT, stationid, stationchannel, location):
	
	# this is specific to acoustic meaurements producing a plot of acoustic power
	# routine can be ignore more non-acoustic readings
	#produces a 24hour obspy 'dayplot' saved in svg format.
	#two copies are produced Today.svg and 'date'.svg
	
	here = os.path.dirname(os.path.realpath(__file__))
	
	Year = str(DailyStartTime.year)
	Month = str(DailyStartTime.month)
	Day = str(DailyStartTime.day)
	yearDir =  'Data' + '/' + Year
	
	StationInfo = str(stationid)+'-'+str(stationchannel)+'-'+str(location)
	
	if (nDailySamples > 3000):
		try:
			tr = CalcRunningMeanPower(st, deltaT, lowCut, highCut)
			Year = str(DailyStartTime.year)
			Month = str(DailyStartTime.month)
			Day = str(DailyStartTime.day)
			saveDir =  'Plots/' 
			dateString = str(Year) + '_' + str(Month) + '_' + str(Day)
			plotTitle= 'AcousticPower ' +':::'+StationInfo+':::'+' '+dateString+'  '+str(lowCut) + '-' + str(highCut) + ' Hz'
			filename1 = ('Plots/TodayAcousticPower.svg')
			filename2 = 'Plots/' + dateString+'__'+StationInfo+'__AcousticPower.svg'
			tr.plot(type="dayplot",outfile=filename1, title=plotTitle, data_unit='$Wm^{-2}$', interval=60, right_vertical_labels=False, 
			color=['k', 'r', 'b', 'g'], show_y_UTC_label=False)
			copyfile('Plots/TodayAcousticPower.svg', filename2)
		except (ValueError,IndexError):
			print('an index error on plotting acoustic power!')
 
###---------------------------ooo0ooo---------------------------
def CalcRunningMeanPower(st, deltaT, lowCut, highCut):
	#routine used to hepl plot acoustic power - can be ignored for non-acoustic
	#applications

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
def SaveAndPlot(DailyReadings, HourlyReadings, DailyStartTime, HourlyStartTime,nDailySamples,nHourlySamples, stationid, stationchannel,location, sampleEndTime, deltaT):
	
	#create hourly stream st and save
	st = createMSeed(HourlyReadings, HourlyStartTime, sampleEndTime, nHourlySamples, stationid, stationchannel, location)
	SaveDataMSeed(st, HourlyStartTime)

	#create daily stream, plot and save 
	st = createMSeed(DailyReadings, DailyStartTime,sampleEndTime, nDailySamples, stationid, stationchannel, location)
	st.filter('bandpass', freqmin=lowCut, freqmax=highCut, corners=4, zerophase=True)
	PlotAcousticPower(st, DailyStartTime, nDailySamples, lowCut, highCut, deltaT, stationid, stationchannel, location)
	PlotDayPlot(st, DailyStartTime, nDailySamples, lowCut, highCut, stationid, stationchannel, location)



###--------------------------Main Body--------------------------
###-                                                           -
###-                                                           -
###--------------------------+++++++++--------------------------

sensor = smbus.SMBus(1)    #used to communicate with i2c sensor
addr = 0x28				   # address of i2c sensor
os.chdir('/home/pi/InfraSound')  #replace with home directory of application


# below are station parameters for your station. see SEED manual for
# more details http://www.fdsn.org/pdf/SEEDManual_V2.4.pdf

#-- station parameters
stationid = 'STARF'
# channel B=broadband 10-80Hz sampling, D=pressure sensor, F=infrasound
stationchannel = 'BDF'  #see SEED format documentation
location = '02'  # 2 digit code to identify specific sensor rig
stationNetwork='IR'
#####network above needs adding to code 

SamplingFreq=40.00
SamplingPeriodInSeconds = 1.00/SamplingFreq
DataFileLengthSeconds=3600*24

deltaT = 5.0  # time interval seconds to calculate running mean for acoustic power

#-----create top level Data and plots directoryies if not already present
here = os.path.dirname(os.path.realpath(__file__))

try:
	os.makedirs('Plots/')
except OSError:
	if not os.path.isdir('Plots/'):
		raise

try:
	os.makedirs('Data/')
except OSError:
	if not os.path.isdir('Data/'):
		raise


###---------------------------ooo0ooo---------------------------
# -- create numpy arrays to store pressure and time data
HourlyTargetSampleNumber=int(SamplingFreq*3600*1.1)
DailyTargetSampleNumber=int(HourlyTargetSampleNumber*24)

DailyReadings=np.zeros(DailyTargetSampleNumber, np.float32) 
HourlyReadings=np.zeros(HourlyTargetSampleNumber, np.float32)
nDailySamples = 0
nHourlySamples = 0

lowCut=0.01         # low cut-off frequency
highCut=10.0        # low cut-off frequency


DailyStartTime=UTCDateTime() #initialise variable
HourlyStartTime=UTCDateTime()
sampleEndTime=UTCDateTime() #initialise variable
lastSaveTime=UTCDateTime() #initialise variable



while 1:
        
    time.sleep(SamplingPeriodInSeconds)
    
    try:
        tempPressure=readPressure()
        DailyReadings[nDailySamples] = tempPressure
        HourlyReadings[nHourlySamples] = tempPressure
        nDailySamples = nDailySamples + 1
        nHourlySamples = nHourlySamples + 1
    except IOError:
        print('read error')
        DailyReadings[nDailySamples] = 0.00 	#default misread value
        HourlyReadings[nHourlySamples] = 0.00	#default misread value
        nDailySamples = nDailySamples + 1
        nHourlySamples = nHourlySamples + 1


    if (UTCDateTime().hour != HourlyStartTime.hour):
        sampleEndTime=UTCDateTime()
        threadSaveAndPlot = Thread(target=SaveAndPlot, args=(DailyReadings, \
        HourlyReadings, DailyStartTime, HourlyStartTime, nDailySamples,\
        nHourlySamples, stationid, stationchannel,location, sampleEndTime, deltaT))
        threadSaveAndPlot.start()
        HourlyStartTime = UTCDateTime()
        HourlyReadings=np.zeros(HourlyTargetSampleNumber, np.float32) #reset data array to zero for new hour
        nHourlySamples = 0

	## the following is used to finalise the last day's 24hr plot
    if (DailyStartTime.day != UTCDateTime().day):
        DailyReadings=np.zeros(DailyTargetSampleNumber, np.float32)  #reset data array to zero for new day
        DailyStartTime=UTCDateTime()
        nDailySamples = 0
        gc.collect


