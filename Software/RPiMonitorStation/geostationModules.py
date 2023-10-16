# ---------------------------ooo0ooo---------------------------
#       Seismic Monitoring Software
#       Ian Robinson
#       http://schoolphysicsprojects.org
#
#
#        requires
#           python3, python3-obspy, matplotlib
#           icp10125- https://github.com/pimoroni/icp10125-python
# ---------------------------Notes---------------------------
#

#
# ---------------------------ooo0ooo---------------------------
from obspy import UTCDateTime, Trace, Stream
from matplotlib import pyplot as plt
from globalvariablesModule import *

from shutil import copyfile
#import copyfile
import numpy as np

import matplotlib
matplotlib.use('Agg')  # prevent use of Xwindows


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# below is for the DLVRF50D1NDCNI3F Amphenol mems differential pressure infrasound monitor station

# SENSOR = smbus.SMBus(1)    #used to communicate with i2c sensor
# ADDR = 0x28                # address of i2c sensor
# # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ---------------------------ooo0ooo---------------------------
def read_from_sensor_icp10125():
    # this function specifically reads data over i2c from a
    # icp10125
    pressure, temperature = device.measure()
    readings = (pressure, temperature)

    return readings

# ---------------------------ooo0ooo---------------------------


def read_from_sensor_DLVRF50D():
    # this function specifically reads data over i2c from a
    # DLVRF50D1NDCNI3F Amphenol mems differential pressure SENSOR

    # this routine will need to be changed for a diffent input device
    # such as a voltage via an a/d convertor
    # note: the return value 'pressure' is a floating point number

    raw_data = bytearray
    raw_data = []

    raw_pressure = int

    z = bytearray
    z = [0, 0, 0, 0]
    z = SENSOR.read_i2c_block_data(ADDR, 0, 4)  # offset 0, 4 bytes

    d_masked = (z[0] & 0x3F)
    raw_data = (d_masked << 8) + z[1]

    raw_pressure = int(raw_data)
    pressure = (((raw_pressure-8192.0)/16384.0) * 250.0 * 1.25)

#   2nd value could be from the temperature sensor - not implemented
    readings = (pressure, 0.0)

    return readings
# ---------------------------ooo0ooo---------------------------


def save_hourly_data_as_mseed(st, channel_no):

    start_date_time = st[0].stats.starttime

    year = str(start_date_time.year)
    month = str(start_date_time.month)
    day = str(start_date_time.day)
    hour = str(start_date_time.hour)
    save_dir = 'Data' + '/' + year + '/' + month + '/' + day

    if channel_no == 0:
        filename = hour + '_' + STATION_INFO_0 + '.mseed'

    if channel_no == 1:
        filename = hour + '_' + STATION_INFO_1 + '.mseed'

#   -------create data Directory Structure if not already present
    here = os.path.dirname(os.path.realpath(__file__))

    try:
        os.makedirs('Data/'+year+'/')
    except OSError:
        if not os.path.isdir('Data/'+year+'/'):
            raise

    try:
        os.makedirs('Data/'+year+'/'+month+'/')
    except OSError:
        if not os.path.isdir('Data/'+year+'/'+month+'/'):
            raise

    try:
        os.makedirs('Data/'+year+'/'+month+'/'+day+'/')
    except OSError:
        if not os.path.isdir('Data/'+year+'/'+month+'/'+day+'/'):
            raise

    filepath = os.path.join(here, save_dir, filename)

    datafile = open(filepath, 'wb')
    st.write(datafile, format='MSEED', encoding=4, reclen=4096)
    datafile.close()


# ---------------------------ooo0ooo---------------------------


def save_weekly_data_as_mseed(st, channel_no):
    weekly_start_time = st[0].stats.starttime
    year = str(weekly_start_time.year)
    month = str(weekly_start_time.month)
    day = str(weekly_start_time.day)

    save_dir = 'Data' + '/' + year + '/' + month + '/'

    if channel_no == 0:
        filename = day + '_' + month + '_' + year + \
            '_weekly_' + STATION_INFO_0 + '.mseed'

    if channel_no == 1:
        filename = day + '_' + month + '_' + year + \
            '_weekly_' + STATION_INFO_1 + '.mseed'


# ----------create data Directory Structure if not already present
    here = os.path.dirname(os.path.realpath(__file__))

    try:
        os.makedirs('Data/'+year+'/')
    except OSError:
        if not os.path.isdir('Data/'+year+'/'):
            raise

    try:
        os.makedirs('Data/'+year+'/'+month+'/')
    except OSError:
        if not os.path.isdir('Data/'+year+'/'+month+'/'):
            raise

    filepath = os.path.join(here, save_dir, filename)

    datafile = open(filepath, 'wb')
    st.write(datafile, format='MSEED', encoding=4, reclen=4096)
    datafile.close()

# ---------------------------ooo0ooo---------------------------


def plot_daily(st, channel_no):

    if st[0].stats.npts > 100:
        try:
            start_time = st[0].stats.starttime
            end_time = st[0].stats.endtime
            year = str(start_time.year)
            month = str(start_time.month)
            day = str(start_time.day)

            save_dir = 'Plots/'
            date_string = str(year) + '-' + str(month) + '-' + str(day)
            start_hour = start_time.hour
            start_minute = start_time.minute
            time_sampled = end_time-start_time  # length of data in seconds
            graph_start = float(start_hour) + float(start_minute/60)
            graph_end = graph_start + (time_sampled/3600.0)
            y_values = st[0].data[3:st[0].stats.npts]
            x_values = np.linspace(graph_start, graph_end, len(y_values))

            updated = UTCDateTime().strftime("%A, %d. %B %Y %I:%M%p")
            averaging_length = int(AVERAGINGTIME * st[0].stats.sampling_rate)

            if channel_no == 0:
                filename1 = ('Plots/today_' + DATATYPE_CHANNEL_0 + '.svg')
                filename2 = ('Plots/' + date_string + '_' +
                             STATION_INFO_0 + '_' + DATATYPE_CHANNEL_0 + '.svg')

                if averaging_length > 0:
                    y_values = moving_average(y_values, averaging_length)

                x_values = np.linspace(graph_start, graph_end, len(y_values))
                # convert to hPa
                y_values = y_values/100.0

                x_values = np.linspace(graph_start, graph_end, len(y_values))
                fig = plt.figure(figsize=(12, 4))
                x_axis_text = ('Updated - ' + updated + ' UTC')
                plt.title(PLOT_TITLE_CHANNEL_0 +
                          date_string + ' : ' + STATION_INFO_0)
                plt.xlabel(x_axis_text)
                # zeroLabel=str('%0.5g' % (zeroPoint))
                plt.ylabel(PLOT_YLABEL_CHANNEL_0)

                plt.plot(x_values, y_values,  marker='None',
                         color='darkolivegreen')
                plt.xlim(0, 24.01)
                plt.xticks(np.arange(0, 24.01, 2.0))
                plt.grid(True)
                plt.savefig(filename1)
                copyfile(filename1, filename2)
                plt.close('all')

            if channel_no == 1:
                filename1 = ('Plots/today_' + DATATYPE_CHANNEL_1 + '.svg')
                filename2 = ('Plots/' + date_string + '_' +
                             STATION_INFO_1 + '_' + DATATYPE_CHANNEL_1 + '.svg')

                if averaging_length > 0:
                    y_values = moving_average(y_values, averaging_length)

                x_values = np.linspace(graph_start, graph_end, len(y_values))

                fig = plt.figure(figsize=(12, 4))
                x_axis_text = ('Updated - ' + updated + ' UTC')
                plt.title(PLOT_TITLE_CHANNEL_1 +
                          date_string + ' : ' + STATION_INFO_1)
                plt.xlabel(x_axis_text)
                # zeroLabel=str('%0.5g' % (zeroPoint))
                plt.ylabel(PLOT_YLABEL_CHANNEL_1)
                plt.plot(x_values, y_values,  marker='None',
                         color='darkolivegreen')
                plt.xlim(0, 24.01)
                plt.xticks(np.arange(0, 24.01, 2.0))
                plt.grid(True)
                plt.savefig(filename1)

                copyfile(filename1, filename2)
                plt.close('all')

        except (ValueError, IndexError):
            print('an  error on plotting daily')


# ---------------------------ooo0ooo---------------------------
def plot_weekly(st, channel_no):

    try:
        start_date_time = st[0].stats.starttime
        start_year = start_date_time.year
        start_month = start_date_time.month
        # start_day as day of month i.e  1, 2 ... 31
        start_day = start_date_time.day
        # start day as numerical day of week i.e.  1 (Mon) to 7 (Sun)
        iso_start_day = start_date_time.isoweekday()
        start_hour = start_date_time.hour
        start_minute = start_date_time.minute

        date_string = str(start_year) + '-' + \
            str(start_month) + '-' + str(start_day)

        time_sampled = st[0].stats.npts / \
            st[0].stats.sampling_rate  # length of data in seconds
        graph_start = float(iso_start_day-1) + \
            (float(start_hour)/24) + float(start_minute/1440)
        graph_end = graph_start + (time_sampled/86400.0)

        updated = UTCDateTime().strftime("%A, %d. %B %Y %I:%M%p")
        date_string = str(start_year) + '-' + \
            str(start_month) + '-' + str(start_day)

        if channel_no == 0:
            filename1 = ('Plots/weekly_' + DATATYPE_CHANNEL_0 + '.svg')
            filename2 = ('Plots/' + date_string + '_weekly_' +
                         STATION_INFO_0 + '_' + DATATYPE_CHANNEL_0 + '.svg')
            filename3 = ('Plots/' + 'prevWeekly_' +
                         STATION_INFO_0 + '_' + DATATYPE_CHANNEL_0 + '.svg')

            y_values = st[0].data[0:(st[0].stats.npts-1)]
            y_values = y_values/100.0  # convert to hPa

            x_values = np.linspace(graph_start, graph_end, len(y_values))

            fig = plt.figure(figsize=(12, 4))

            x_axis_text = ("Updated - " + updated + " UTC")
            plt.title(PLOT_TITLE_CHANNEL_0 +
                      date_string + ' : ' + STATION_INFO_0)

            plt.xlabel(x_axis_text)
            plt.ylabel(PLOT_YLABEL_CHANNEL_0)

            plt.plot(x_values, y_values,  marker='None',
                     color='darkolivegreen')
            plt.xlim(0.0, 7.01)

            ticks = np.arange(0, 7, 1.0)
            labels = "Mon Tues Weds Thurs Fri Sat Sun".split()
            plt.xticks(ticks, labels)
            plt.grid(True)

            plt.savefig(filename1)
            copyfile(filename1, filename2)
            plt.close('all')

            # midnight sunday-monday  - save previous weekly plot with datestamped name
            time_now = UTCDateTime()

            if time_now.isoweekday() == 1:
                if time_now.hour == 0:
                    copyfile(filename1, filename3)
                    print('prev week saved')

        if channel_no == 1:
            filename1 = ('Plots/weekly_' + DATATYPE_CHANNEL_1 + '.svg')
            filename2 = ('Plots/' + date_string + '_weekly_' +
                         STATION_INFO_1 + '_' + DATATYPE_CHANNEL_1 + '.svg')
            filename3 = ('Plots/' + 'prevWeekly_' +
                         STATION_INFO_1 + '_' + DATATYPE_CHANNEL_1 + '.svg')

            y_values = st[0].data[0:(st[0].stats.npts-1)]

            x_values = np.linspace(graph_start, graph_end, len(y_values))

            fig = plt.figure(figsize=(12, 4))

            x_axis_text = ("Updated - " + updated + " UTC")
            plt.title(PLOT_TITLE_CHANNEL_1 +
                      date_string + ' : ' + STATION_INFO_1)

            plt.xlabel(x_axis_text)
            plt.ylabel(PLOT_YLABEL_CHANNEL_1)

            plt.plot(x_values, y_values,  marker='None',
                     color='darkolivegreen')
            plt.xlim(0.0, 7.01)

            ticks = np.arange(0, 7, 1.0)
            labels = "Mon Tues Weds Thurs Fri Sat Sun".split()
            plt.xticks(ticks, labels)
            plt.grid(True)

            plt.savefig(filename1)
            copyfile(filename1, filename2)
            plt.close('all')

            # midnight sunday-monday  - save previous weekly plot with datestamped name
            time_now = UTCDateTime()

            if time_now.isoweekday() == 1:
                if time_now.hour == 0:
                    copyfile(filename1, filename3)
                    print('prev week saved')
    except (ValueError, IndexError):
        print('an error on plotting weekly data')


# ---------------------------ooo0ooo---------------------------


def plotPrev168hrs(tmp_prev_168hr_data, start_time_prev168hrs_data,
                   end_time_prev168hrs_data, ntmp_prev_168hr_data, channel_no):
    try:

        sample_end_minus_168_hrs = end_time_prev168hrs_data - (168*3600)

        # our plot area runs from the start of the day, need to determine day of week and
        # time offset from start of day to sample_end_minus_168_hrs

        # start day as numerical day of week i.e.  1 (Mon) to 7 (Sun)
        plot_start_day = sample_end_minus_168_hrs.isoweekday()

        # determine number of seconds since start of day
        offset_from_day_start = (end_time_prev168hrs_data.hour * 3600) + \
            (end_time_prev168hrs_data.minute * 60) + \
            (end_time_prev168hrs_data.second)

        # convert sampleStart Datetime to offset in minutes from **plot** start
        datastart_minute = int(
            (start_time_prev168hrs_data - sample_end_minus_168_hrs + offset_from_day_start) / 60.0)

        datastart_minute = max(datastart_minute, 0)  # if <0 set to 0

        data_end_minute = int(datastart_minute + ntmp_prev_168hr_data)

        # determine date info for graph header
        start_year = start_time_prev168hrs_data.year
        start_month = start_time_prev168hrs_data.month

        # start_day as day of month i.e  1, 2 ... 31
        start_day = start_time_prev168hrs_data.day

        date_string = str(start_year) + '-' + \
            str(start_month) + '-' + str(start_day)

        updated = UTCDateTime().strftime("%A, %d. %B %Y %I:%M%p")

        if channel_no == 0:
            filename1 = ('Plots/prev168hrs_' + 'acoustic_pwr' + '.svg')
            y_values = tmp_prev_168hr_data[0:(ntmp_prev_168hr_data-1), 0]
            # y_values = y_values/100.0  # convert to hPa

            x_values = np.linspace(
                datastart_minute, data_end_minute, len(y_values))

            fig = plt.figure(figsize=(12, 4))

            x_axis_text = ("Updated - " + updated + " UTC")
            # plt.title('prev 168 hrs ' + PLOT_TITLE_CHANNEL_0_b +
            #           date_string + ' : ' + STATION_INFO_0)
            plt.title('prev 168 hrs Acoustic Pwr' + ' : ' + STATION_INFO_0 +
                      ' : ' + str(FRQ_LOW_CUT) + '-' + str(FRQ_HIGH_CUT) + ' Hz')
            plt.xlabel(x_axis_text)
            plt.ylabel(PLOT_YLABEL_CHANNEL_0)
            plt.plot(x_values, y_values,  marker='None',
                     color='darkolivegreen')

            # xAxis 1 minute per division
            # therefore 8 days = 8*24*60 = 11520 minutes
            plt.xlim(0.0, 11520.0)
            ticks = np.arange(0.0, 11520, 1440.0)  # daily ticks

            # create labels for ticks
            days = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']
            labels = []
            for i in range(0, 8):
                j = (plot_start_day - 1 + i) % 7
                labels.append(days[j])

            plt.xticks(ticks, labels)
            plt.grid(True)
            plt.savefig(filename1)
            plt.close('all')

        if channel_no == 1:
            filename1 = ('Plots/prev168hrs_' + DATATYPE_CHANNEL_1 + '.svg')
            y_values = tmp_prev_168hr_data[0:(ntmp_prev_168hr_data-1), 1]

            x_values = np.linspace(
                datastart_minute, data_end_minute, len(y_values))

            fig = plt.figure(figsize=(12, 4))

            x_axis_text = ("Updated - " + updated + " UTC")
            plt.title('prev 168 hrs ' + PLOT_TITLE_CHANNEL_1 +
                      date_string + ' : ' + STATION_INFO_1)
            plt.xlabel(x_axis_text)
            plt.ylabel(PLOT_YLABEL_CHANNEL_1)
            plt.plot(x_values, y_values,  marker='None',
                     color='darkolivegreen')

            # xAxis 1 minute per division
            # therefore 8 days = 8*24*60 = 11520 minutes
            plt.xlim(0.0, 11520.0)
            ticks = np.arange(0.0, 11520, 1440.0)  # daily ticks

            # create labels for ticks
            days = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']
            labels = []
            for i in range(0, 8):
                j = (plot_start_day - 1 + i) % 7
                labels.append(days[j])

            plt.xticks(ticks, labels)
            plt.grid(True)
            plt.savefig(filename1)
            plt.close('all')

    except (ValueError, IndexError):
        print('an error on plotting prev 168 hrs')

# ---------------------------ooo0ooo---------------------------
def plot_daily_raw_pressures(st):
    # specifically used by infrasound monitor
    # produces a 24hour obspy 'dayplot' saved in svg format.
    # two copies are produced Today.svg and 'date'.svg

    if (st[0].stats.npts > 1000):
        try:
            here = os.path.dirname(os.path.realpath(__file__))
            start_time = st[0].stats.starttime
            year = str(start_time.year)
            month = str(start_time.month)
            day = str(start_time.day)
            year_dir = 'Data' + '/' + year

            save_dir = 'Plots/'
            date_string = str(year) + '_' + str(month) + '_' + str(day)
            plot_title = PLOT_TITLE_CHANNEL_0 + ' : ' + STATION_INFO_0 + ' : ' + \
                date_string + ' : ' + \
                str(FRQ_LOW_CUT) + '-' + str(FRQ_HIGH_CUT) + ' Hz'
            filename1 = ('Plots/today_raw_pressure.svg')
            filename2 = 'Plots/' + date_string+'__' + STATION_INFO_0 + '__Raw_Pressure.svg'

            st.plot(type="dayplot", outfile=filename1, title=plot_title, data_unit='$\Delta$Pa', interval=60,
                    right_vertical_labels=False, one_tick_per_line=False, color=['k', 'r', 'b', 'g'], show_y_UTC_label=False)
            copyfile(filename1, filename2)
            plt.close('all')
            print('Plotting of daily raw pressure completed')
        except (ValueError, IndexError):
            print('an  error on plotting dayly raw pressures!')

    return None
# ---------------------------ooo0ooo---------------------------


def plot_daily_pwr(st):
    # this is specific to acoustic meaurements producing a plot of acoustic pwr
    # routine can be ignored for non-acoustic readings
    # produces a 24hour obspy 'dayplot' saved in svg format.
    # two copies are produced

    if (st[0].stats.npts > 3000):
        try:
            here = os.path.dirname(os.path.realpath(__file__))

            start_time = st[0].stats.starttime
            year = str(start_time.year)
            month = str(start_time.month)
            day = str(start_time.day)
            year_dir = 'Data' + '/' + year

            tr = calc_running_mean_pwr(st)
            save_dir = 'Plots/'
            date_string = str(year) + '_' + str(month) + '_' + str(day)
            plot_title = 'Acoustic Pwr' + ' : ' + STATION_INFO_0 + ' : ' + \
                date_string + ' : ' + \
                str(FRQ_LOW_CUT) + '-' + str(FRQ_HIGH_CUT) + ' Hz'
            filename1 = ('Plots/today_acoustic_pwr.svg')
            filename2 = 'Plots/' + date_string+'__' + STATION_INFO_0 + '__acoustic_pwr.svg'
            tr.plot(type="dayplot", outfile=filename1, title=plot_title,
                    data_unit='$Wm^{-2}$', interval=60, right_vertical_labels=False, color=['k', 'r', 'b', 'g'], show_y_UTC_label=False)
            copyfile(filename1, filename2)
            plt.close('all')
            print('Plotting of daily acoustic power completed')
        except (ValueError, IndexError):
            print('an error on plotting acoustic pwr!')
    return None

# ---------------------------ooo0ooo---------------------------
def plot_weekly_accoustic_power(st):
    # this is specific to acoustic meaurements producing a plot of acoustic pwr
    # routine can be ignored for non-acoustic readings
    # two copies are produced
    try:
        start_date_time = st[0].stats.starttime
        n_samples = st[0].stats.npts
        SAMPLING_FRQ = st[0].stats.sampling_rate

        start_year = start_date_time.year
        start_month = start_date_time.month
        # start_day as day of month i.e  1, 2 ... 31
        start_day = start_date_time.day
        # start day as numerical day of week i.e.  1 (Mon) to 7 (Sun)
        iso_start_day = start_date_time.isoweekday()
        start_hour = start_date_time.hour
        start_minute = start_date_time.minute

        date_string = str(start_year) + '-' + \
            str(start_month) + '-' + str(start_day)
        filename1 = ('Plots/weekly_acoustic_pwr.svg')
        filename2 = ('Plots/' + date_string + '_weekly_acoustic_pwr.svg')

        time_sampled = n_samples/SAMPLING_FRQ  # length of data in seconds
        graph_start = float(iso_start_day-1) + \
            (float(start_hour)/24) + float(start_minute/1440)
        graph_end = graph_start + (time_sampled/86400.0)

        z = UTCDateTime()
        updated = z.strftime("%A, %d. %B %Y %I:%M%p")
        date_string = str(start_year) + '-' + \
            str(start_month) + '-' + str(start_day)

        weekly_acoustic_pwr = st[0].data
        y_values = weekly_acoustic_pwr[0:(n_samples-1)]
        x_values = np.linspace(graph_start, graph_end, len(y_values))

        fig = plt.figure(figsize=(12,4))

        xAxisText = ("Time  updated - " + updated + " UTC")
        plt.title('Acoustic Pwr - w/c - ' + date_string  + ' : ' + \
                  STATION_INFO_0  + ' : ' + str(FRQ_LOW_CUT) + '-' + str(FRQ_HIGH_CUT) + ' Hz')
        plt.xlabel(xAxisText)
        plt.ylabel('$Wm^{-2}$')

        plt.plot(x_values, y_values,  marker='None',    color='darkolivegreen')
        plt.xlim(0.0, 7.01)

        ticks = np.arange(0, 7, 1.0)
        labels = "Mon Tues Weds Thurs Fri Sat Sun".split()
        plt.xticks(ticks, labels)
        plt.grid(True)
        #plt.fill_between(x_values, y_values)

        plt.savefig(filename1)
        copyfile(filename1, filename2)
        plt.close('all')
        print('Plotting of weekly acoustic power completed')

    except (ValueError, IndexError):
        print('an error on plotting weekly acoustic pwr!')

    return None
# ---------------------------ooo0ooo---------------------------


def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
# ---------------------------ooo0ooo---------------------------

# ---------------------------ooo0ooo---------------------------


def calc_running_mean_pwr(st):
    # routine used to help plot acoustic pwr - can be ignored for non-acoustic
    # applications

    N = len(st[0].data)
    delta_t = st[0].stats.delta
    new_stream = st[0].copy()

    new_stream.filter('bandpass', freqmin=FRQ_LOW_CUT,
                      freqmax=FRQ_HIGH_CUT, corners=4, zerophase=True)
    x = new_stream.data

    x = x**2

    n_sample_points = int(ACC_PWR_INTERVAL/delta_t)
    running_mean = np.zeros((N-n_sample_points), np.float32)

    # determine first tranche
    temp_sum = 0.0

    for i in range(0, (n_sample_points-1)):
        temp_sum = temp_sum + x[i]

        running_mean[i] = temp_sum/n_sample_points

    # calc rest of the sums by subtracting first value and adding new one from far end
    for i in range(1, (N-(n_sample_points+1))):
        temp_sum = temp_sum - x[i-1] + x[i + n_sample_points]
        running_mean[i] = temp_sum/n_sample_points
    # calc averaged acoustic intensity as P^2/(density*c)
    running_mean = running_mean/(1.2*330)

    new_stream.data = running_mean
    new_stream.stats.npts = len(running_mean)

    return new_stream

# ---------------------------ooo0ooo---------------------------


def calc_acoustic_pwr(tranche, n_sample_points):

    tranche = tranche**2
    temp_sum = 0.0
    if (n_sample_points > 1):
        for i in range(0, (n_sample_points-1)):
            temp_sum = temp_sum+tranche[i]
    else:
        print("tranche too small = ", n_sample_points)

    mean = temp_sum/n_sample_points
# calc averaged acoustic intensity as P^2/(density*c)
    acoustic_pwr = mean/(1.2*330)
    #print("accoustic power", acoustic_pwr)

    return acoustic_pwr
# ---------------------------ooo0ooo---------------------------


def create_mseed(readings, start_time, end_time, n_samples, channel_no):

    true_sample_frequency = float(n_samples) / (end_time - start_time)

    # set current time

    # Fill header attributes
    if channel_no == 0:
        stats = {'network': STATION_NETWORK, 'station': STATION_ID, 'location': STATION_LOCATION,
                 'channel': STATION_CHANNEL_0, 'npts': n_samples, 'sampling_rate': true_sample_frequency,
                 'mseed': {'dataquality': 'D'}}
        stats['starttime'] = start_time
        stats['endtime'] = end_time
        st = Stream([Trace(data=readings[0:n_samples, 0], header=stats)])

    if channel_no == 1:
        stats = {'network': STATION_NETWORK, 'station': STATION_ID, 'location': STATION_LOCATION,
                 'channel': STATION_CHANNEL_1, 'npts': n_samples, 'sampling_rate': true_sample_frequency,
                 'mseed': {'dataquality': 'D'}}
        stats['starttime'] = start_time
        stats['endtime'] = end_time
        st = Stream([Trace(data=readings[0:n_samples, 1], header=stats)])

    return st


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
