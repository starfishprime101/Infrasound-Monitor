# global Constants


# import below is for the icp10125 based microbarometer station
#from icp10125 import ICP10125, ULTRA_LOW_NOISE
#device = ICP10125()

import os
import smbus

home_directory = '/home/ian/GeoPhysics'
# replace with home directory of application


# below is for the DLVRF50D1NDCNI3F Amphenol mems differential pressure infrasound monitor station
SENSOR = smbus.SMBus(1)  # used to communicate with i2c sensor
ADDR = 0x28                # address of i2c sensor


# ---------------------------ooo0ooo---------------------------
# below are station parameters for your station. see SEED manual for
# more details http://www.fdsn.org/pdf/SEEDManual_V2.4.pdf  esp. Apprendix A

MULTIPLE_SENSORS = False   # only channel 0 will be used if false
# -- station parameters
# change this to your sensor # channel M=mid-period 1-10Hz sampling, D=pressure sensor, O = outside
# change this to your sensor id # channel M=mid-period 1-10Hz sampling, K=temperature sensor, O = outside
STATION_ID = 'STARF'
STATION_CHANNEL_0 = 'BDF'  # see SEED format documentation
STATION_CHANNEL_1 = 'MKO'  # see SEED format documentation
STATION_LOCATION = '03'  # 2 digit code to identify specific sensor rig
STATION_NETWORK = 'IR'
STATION_INFO_0 = STATION_ID + '-' + STATION_CHANNEL_0 + '-' + STATION_LOCATION
STATION_INFO_1 = STATION_ID + '-' + STATION_CHANNEL_1 + '-' + STATION_LOCATION

# filesnames for plots =
DATATYPE_CHANNEL_0 = 'Differential Pressure'
PLOT_TITLE_CHANNEL_0 = 'Differential Pressure, Guisborough, UK '
PLOT_TITLE_CHANNEL_0_b = 'Acoustic Power, Guisborough, UK '
PLOT_YLABEL_CHANNEL_0 = 'Acoustic Power W/m^2'

DATATYPE_CHANNEL_1 = 'temperature'
PLOT_TITLE_CHANNEL_1 = 'Temperature, Shed, Guisborough UK '
PLOT_YLABEL_CHANNEL_1 = 'Temperature Celcius'

SAMPLING_FRQ = 70.0
SAMPLING_PERIOD = 1.00/SAMPLING_FRQ
AVERAGINGTIME = 30  # time interval seconds to calculate running mean

N_TARGET_HOURLY_SAMPLES = int(SAMPLING_FRQ*3600*1.5)
N_TARGET_DAILY_SAMPLES = int(N_TARGET_HOURLY_SAMPLES * 24)
N_TARGET_PREV_MINUTE = int(SAMPLING_FRQ*60*1.3)

# no of weekly samples depends on averaging period e.g. 1 per minute ~10080
N_TARGET_WEEKLY_SAMPLES = 12000

# specific to infrasound monitor
ACC_PWR_INTERVAL = 0.1  # time interval seconds to calculate running mean for acoustic pwr
FRQ_LOW_CUT = 0.01       # low freq filter cut off used when plotting
FRQ_HIGH_CUT = 20.0       # high freq filter cut off used when plotting
