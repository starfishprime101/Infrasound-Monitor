	
# ---------------------------ooo0ooo---------------------------
# import required modules
import numpy as np
import time
import array
import multiprocessing
import obspy

from array import array
from datetime import datetime
from datetime import timedelta

from scipy import signal

from obspy.imaging.spectrogram import spectrogram
from obspy.signal.filter import bandpass, lowpass, highpass
from obspy.imaging.cm import obspy_sequential
from obspy.signal.tf_misfit import cwt
from obspy import UTCDateTime, read, Trace, Stream
from obspy.signal.trigger import plot_trigger, z_detect
from obspy.io.xseed import Parser
from obspy.signal import PPSD


import time
import string

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

import matplotlib.dates as mdates
import statistics  # used by median filter
import os
import gc
from tkinter import filedialog
from tkinter import *
from tkinter import filedialog
# from tkinter import * #for file opening dialog
import tkinter as tk


# ---------------------------ooo0ooo---------------------------
# ---------------------------ooo0ooo---------------------------

def plotFiltered(Data, samplingFreq, startDT):
    N = len(Data)
    mean_removed = np.ones_like(Data) * np.mean(Data)
    # Data0 = Data - mean_removed
    Data0 = Data

    lowCut1 = 0.001
    highCut1 = 2.0
    Data1 = butter_bandpass_filter(Data0, lowCut1, highCut1, samplingFreq, order=3)
    # legend1=

    lowCut2 = 2.0
    highCut2 = 5.0
    Data2 = butter_bandpass_filter(Data0, lowCut2, highCut2, samplingFreq, order=3)

    lowCut3 = 05.0
    highCut3 = 10.0
    Data3 = butter_bandpass_filter(Data0, lowCut3, highCut3, samplingFreq, order=3)

    xN = np.linspace(1, (N / samplingFreq), N)
    x = np.divide(xN, 60)
    TitleString = ('A: Raw  B:' + str(lowCut1) + '-' + str(highCut1) + ' Hz  C:' + str(lowCut2) + '-' + str(
        highCut2) + ' Hz  D:' + str(lowCut3) + '-' + str(highCut3) + ' Hz')

    fig = plt.figure()
    fig.canvas.set_window_title('start U.T.C. - ' + startDT)

    plt.subplots_adjust(hspace=0.001)
    gs = gridspec.GridSpec(4, 1)

    ax0 = plt.subplot(gs[0])
    ax0.plot(x, Data0, label='unfiltered')

    ax1 = plt.subplot(gs[1], sharex=ax0)
    ax1.plot(x, Data1, label=(str(lowCut1) + '-' + str(highCut1) + 'Hz'))

    ax2 = plt.subplot(gs[2], sharex=ax0)
    ax2.plot(x, Data2, label='unfiltered')

    ax3 = plt.subplot(gs[3], sharex=ax0)
    ax3.plot(x, Data3, label='unfiltered')

    xticklabels = ax0.get_xticklabels() + ax1.get_xticklabels() + ax2.get_xticklabels()
    plt.setp(xticklabels, visible=False)

    fig.tight_layout()
    fig.show()

    # plt.show()
    # ---------------------------ooo0ooo---------------------------


def plotRawSignalBands(tr):
    print('Filtering and plotting raw signal bands')

    legendLoc = 'upper right'
    #xMin = 0.0
    #xMax = 24.0

    N = len(tr.data)

    samplingFreq = tr.stats.sampling_rate

    yscale = 5.0

    lowCut1 = 0.01
    highCut1 = 0.03
    tr1 = tr.copy()
    tr1.filter('bandpass', freqmin=lowCut1, freqmax=highCut1, corners=4, zerophase=True)

    lowCut2 = 0.03
    highCut2 = 0.06
    tr2 = tr.copy()
    tr2.filter('bandpass', freqmin=lowCut2, freqmax=highCut2, corners=4, zerophase=True)

    lowCut3 = 0.06
    highCut3 = 0.08
    tr3 = tr.copy()
    tr3.filter('bandpass', freqmin=lowCut3, freqmax=highCut3, corners=4, zerophase=True)

    lowCut4 = 0.08
    highCut4 = 0.1
    tr4 = tr.copy()
    tr4.filter('bandpass', freqmin=lowCut4, freqmax=highCut4, corners=4, zerophase=True)

    lowCut5 = 0.1
    highCut5 = 0.15
    tr5 = tr.copy()
    tr5.filter('bandpass', freqmin=lowCut5, freqmax=highCut5, corners=5, zerophase=True)

    lowCut6 = 0.15
    highCut6 = 0.2
    tr6 = tr.copy()
    tr6.filter('bandpass', freqmin=lowCut6, freqmax=highCut6, corners=4, zerophase=True)

    lowCut7 = 0.2
    highCut7 = 0.25
    tr7 = tr.copy()
    tr7.filter('bandpass', freqmin=lowCut7, freqmax=highCut7, corners=4, zerophase=True)

    x = np.linspace(1, (N / tr.stats.sampling_rate), N)
    x = np.divide(x, 3600)

    fig = plt.figure()
    fig.suptitle(str(tr.stats.starttime) + ' Filtered ')
    fig.canvas.set_window_title('start U.T.C. - ' + str(tr.stats.starttime))

    plt.subplots_adjust(hspace=0.001)
    gs = gridspec.GridSpec(8, 1)

    ax0 = plt.subplot(gs[0])
    ax0.plot(x, tr)
    #ax0.set_xlim(xMin, xMax)
    ax0.legend(['raw data'], loc=legendLoc, fontsize=10)

    ax1 = plt.subplot(gs[1], sharex=ax0)
    ax1.plot(x, tr1)
    ax1.legend([str(lowCut1) + '-' + str(highCut1) + 'Hz'], loc=legendLoc, fontsize=10)

    ax2 = plt.subplot(gs[2], sharex=ax1)
    ax2.plot(x, tr2)
    ax2.legend([str(lowCut2) + '-' + str(highCut2) + 'Hz'], loc=legendLoc, fontsize=10)

    ax3 = plt.subplot(gs[3], sharex=ax1)
    ax3.plot(x, tr3)
    ax3.legend([str(lowCut3) + '-' + str(highCut3) + 'Hz'], loc=legendLoc, fontsize=10)

    ax4 = plt.subplot(gs[4], sharex=ax1)
    ax4.plot(x, tr4)
    ax4.legend([str(lowCut4) + '-' + str(highCut4) + 'Hz'], loc=legendLoc, fontsize=10)

    ax5 = plt.subplot(gs[5], sharex=ax1)
    ax5.plot(x, tr5)
    ax5.legend([str(lowCut5) + '-' + str(highCut5) + 'Hz'], loc=legendLoc, fontsize=10)

    ax6 = plt.subplot(gs[6], sharex=ax1)
    ax6.plot(x, tr6)
    ax6.legend([str(lowCut6) + '-' + str(highCut6) + 'Hz'], loc=legendLoc, fontsize=10)

    ax7 = plt.subplot(gs[7], sharex=ax1)
    ax7.plot(x, tr7)
    ax7.legend([str(lowCut7) + '-' + str(highCut7) + 'Hz'], loc=legendLoc, fontsize=10)

    xticklabels = ax0.get_xticklabels() + ax1.get_xticklabels() + ax2.get_xticklabels() + ax3.get_xticklabels() \
                  + ax4.get_xticklabels() + ax5.get_xticklabels() + ax6.get_xticklabels()

    plt.setp(xticklabels, visible=False)

    ax7.set_xlabel(r'$\Delta$t - hr', fontsize=12)

    fig.tight_layout()
    fig.show()
    input("Press Enter to continue...")

    # ---------------------------ooo0ooo---------------------------


def plotAcousticBands(tr, deltaT):
    print('Filtering and plotting acoustic bands')

    legendLoc = 'upper right'
    #xMin = 0.0
    #xMax = 24.0



    samplingFreq = tr.stats.sampling_rate
    timeInsecs= len(tr.data)/samplingFreq


    lowCut0 = 0.01
    highCut0 = 15.0
    tr0 = tr.copy()
    tr0.filter('bandpass', freqmin=lowCut0, freqmax=highCut0, corners=4, zerophase=True)
    tr0b = CalcRunningMeanPower(tr0, deltaT)
    

    lowCut1 = 0.01
    highCut1 = 0.1
    tr1 = tr.copy()
    tr1.filter('bandpass', freqmin=lowCut1, freqmax=highCut1, corners=4, zerophase=True)
    tr1b = CalcRunningMeanPower(tr1, deltaT)

    lowCut2 = 0.1
    highCut2 = 0.5
    tr2 = tr.copy()
    tr2.filter('bandpass', freqmin=lowCut2, freqmax=highCut2, corners=4, zerophase=True)
    tr2b = CalcRunningMeanPower(tr2, deltaT)

    lowCut3 = 0.5
    highCut3 = 1.0
    tr3 = tr.copy()
    tr3.filter('bandpass', freqmin=lowCut3, freqmax=highCut3, corners=4, zerophase=True)
    tr3b = CalcRunningMeanPower(tr1, deltaT)

    lowCut4 = 1.0
    highCut4 = 2.0
    tr4 = tr.copy()
    tr4.filter('bandpass', freqmin=lowCut4, freqmax=highCut4, corners=4, zerophase=True)
    tr4b = CalcRunningMeanPower(tr4, deltaT)

    lowCut5 = 2.0
    highCut5 = 5.0
    tr5 = tr.copy()
    tr5.filter('bandpass', freqmin=lowCut5, freqmax=highCut5, corners=5, zerophase=True)
    tr5b = CalcRunningMeanPower(tr5, deltaT)

    lowCut6 = 5.0
    highCut6 = 10.0
    tr6 = tr.copy()
    tr6.filter('bandpass', freqmin=lowCut6, freqmax=highCut6, corners=4, zerophase=True)
    tr6b = CalcRunningMeanPower(tr6, deltaT)

    lowCut7 = 10.0
    highCut7 = 15.0
    tr7 = tr.copy()
    tr7.filter('bandpass', freqmin=lowCut7, freqmax=highCut7, corners=4, zerophase=True)
    tr7b = CalcRunningMeanPower(tr7, deltaT)

    N = len(tr0b.data)
    x = np.linspace(1, timeInsecs, N)
    x = np.divide(x, 3600)

    fig = plt.figure()
    fig.suptitle(str(tr.stats.starttime) + 'Acoustic Power Bands')
    fig.canvas.set_window_title('start U.T.C. - ' + str(tr.stats.starttime))

    plt.subplots_adjust(hspace=0.001)
    gs = gridspec.GridSpec(8, 1)

    ax0 = plt.subplot(gs[0])
    ax0.plot(x, tr0b)
    #ax0.set_xlim(xMin, xMax)
    #ax0.set_yscale("log", nonposy='clip')
    ax0.legend(['raw data'], loc=legendLoc, fontsize=10)

    ax1 = plt.subplot(gs[1], sharex=ax0)
    ax1.plot(x, tr1b)
    ax1.legend([str(lowCut1) + '-' + str(highCut1) + 'Hz'], loc=legendLoc, fontsize=10)

    ax2 = plt.subplot(gs[2], sharex=ax1)
    ax2.plot(x, tr2b)
    ax2.legend([str(lowCut2) + '-' + str(highCut2) + 'Hz'], loc=legendLoc, fontsize=10)

    ax3 = plt.subplot(gs[3], sharex=ax1)
    ax3.plot(x, tr3b)
    ax3.legend([str(lowCut3) + '-' + str(highCut3) + 'Hz'], loc=legendLoc, fontsize=10)

    ax4 = plt.subplot(gs[4], sharex=ax1)
    ax4.plot(x, tr4b)
    ax4.legend([str(lowCut4) + '-' + str(highCut4) + 'Hz'], loc=legendLoc, fontsize=10)

    ax5 = plt.subplot(gs[5], sharex=ax1)
    ax5.plot(x, tr5b)
    ax5.legend([str(lowCut5) + '-' + str(highCut5) + 'Hz'], loc=legendLoc, fontsize=10)

    ax6 = plt.subplot(gs[6], sharex=ax1)
    ax6.plot(x, tr6b)
    ax6.legend([str(lowCut6) + '-' + str(highCut6) + 'Hz'], loc=legendLoc, fontsize=10)

    ax7 = plt.subplot(gs[7], sharex=ax1)
    ax7.plot(x, tr7b)
    ax7.legend([str(lowCut7) + '-' + str(highCut7) + 'Hz'], loc=legendLoc, fontsize=10)

    xticklabels = ax0.get_xticklabels() + ax1.get_xticklabels() + ax2.get_xticklabels() + ax3.get_xticklabels() \
                  + ax4.get_xticklabels() + ax5.get_xticklabels() + ax6.get_xticklabels()

    plt.setp(xticklabels, visible=False)

    ax7.set_xlabel(r'$\Delta$t - hr', fontsize=12)

    fig.tight_layout()
    fig.show()
    input("Press Enter to continue...")
    # ---------------------------ooo0ooo---------------------------


# ---------------------------ooo0ooo---------------------------
def butter_bandpass(lowcut, highCut, samplingFreq, order=5):
    nyq = 0.5 * samplingFreq
    low = lowcut / nyq
    high = highCut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


# ---------------------------ooo0ooo---------------------------

def butter_bandpass_filter(data, lowcut, highCut, samplingFreq, order=5):
    b, a = butter_bandpass(lowcut, highCut, samplingFreq, order=order)
    y = lfilter(b, a, data)
    return y


# ---------------------------ooo0ooo---------------------------

def plotPeriodogram(tr, fMin, fMax):
    print('plotting Welch Periodogram....')
    Data = tr.data
    samplingFreq = tr.stats.sampling_rate

    N = len(Data)  # Number of samplepoints
    xN = np.linspace(1, (N / samplingFreq), N)
    # xN = np.divide(xN, 60)
    t1 = np.divide(xN, (samplingFreq * 60))

    x0 = 0
    x1 = N - 2
    if (x0 < 0):
        x0 = 0
    if (x1 > N):
        x1 = N - 1
    subSetLen = x1 - x0

    WINDOW_LEN = int((subSetLen / samplingFreq) * 1)
    OVERLAP_LEN = WINDOW_LEN / 8

    topX = np.linspace((x0 / samplingFreq) + 1, (x1 / samplingFreq), subSetLen)

    f, Pxx = signal.welch(Data[x0:x1], samplingFreq, nperseg=2000, scaling='density')
    Pxx = np.divide(Pxx, 60)

    fig, ax = plt.subplots()

    ax.set_title("Welch Power Density Log Spectrum")
    ax.grid()

    ax.semilogy(f, Pxx)
    ax.set_xlim([0.0, fMax - 2.0])
    ax.set_xlabel(r'f - Hz', fontsize=14)
    ax.set_ylabel(r'Relative Power Amplitude', fontsize=14)

    fig.tight_layout()
    fig.savefig("Welch.png")
    fig.show()

    plt.show()


# ---------------------------ooo0ooo---------------------------
def plotSpectrogram(tr, lowCut, highCut):
    print('plotting Spectrogram....')

    tr.spectrogram(log=True)

    samplingFreq = tr.stats.sampling_rate

    N = len(Data)  # Number of samplepoints
    xN = np.linspace(1, (N / samplingFreq), N)
    # xN = np.divide(xN, 60)
    t1 = np.divide(xN, (samplingFreq * 60))

    fig = plt.figure()
    fig.canvas.set_window_title('FFT Spectrum ' + str(tr.stats.starttime.date))

    ax1 = fig.add_axes([0.1, 0.75, 0.7, 0.2])  # [left bottom width height]
    ax2 = fig.add_axes([0.1, 0.1, 0.7, 0.60], sharex=ax1)
    ax3 = fig.add_axes([0.83, 0.1, 0.03, 0.6])
    # ax2.set_ylim([1.0,3.0])

    #  t = np.arange(spl1[0].stats.N) / spl1[0].stats.sampling_rate
    ax1.plot(xN, Data, 'k')

    # ax,spec = spectrogram(Data, samplingFreq, show=False, axes=ax2)

    ax = spectrogram(Data, samplingFreq, show=False, axes=ax2)
    mappable = ax2.images[0]
    plt.colorbar(mappable=mappable, cax=ax3)

    ax1.set_ylabel(r'$\Delta$ P - Pa')
    ax2.set_ylabel(r'f - Hz', fontsize=14)
    ax2.set_xlabel(r'$\Delta$t - s', fontsize=12)
    ax2.set_ylim([0.0, 6])
    # ax2.set_ybound(lower=None, upper=(highCut*2.0))
    ax2.set_ybound(lower=None, upper=(highCut))

    fig.show()
    return


# ---------------------------ooo0ooo---------------------------
def plotWaveletTransform(tr1, f_min, f_max):
    print('Calculating Wavelet Transform')
    N = len(tr1.data)  # Number of samplepoints
    dt = tr1.stats.delta

    x0 = 0
    x1 = N - 1

    t = np.linspace(x0, x1, num=N)
    t1 = np.divide(t, (tr1.stats.sampling_rate * 60))

    fig = plt.figure()
    fig.suptitle('Wavelet Transform ' + str(tr1.stats.starttime.date), fontsize=12)
    fig.canvas.set_window_title('Wavelet Transform ' + str(tr1.stats.starttime.date))
    # ax2 = fig.add_axes([0.1, 0.1, 0.7, 0.60])
    ax1 = fig.add_axes([0.1, 0.75, 0.7, 0.2])  # [left bottom width height]
    ax2 = fig.add_axes([0.1, 0.1, 0.7, 0.60], sharex=ax1)

    print("x1", x1, "len t", len(t), "len t1", len(t1))

    ax1.plot(t1, tr1.data, 'k')
    ax1.set_ylabel(r'$\Delta$P - Pa')

    scalogram = cwt(tr1.data[x0:x1], dt, 8, f_min, f_max)

    x, y = np.meshgrid(t1, np.logspace(np.log10(f_min), np.log10(f_max), scalogram.shape[0]))

    ax2.pcolormesh(x, y, np.abs(scalogram), cmap=obspy_sequential)

    ax2.set_xlabel("Time  [min]")
    ax2.set_ylabel("Frequency [Hz]")
    ax2.set_yscale('log')
    ax2.set_ylim(f_min, f_max)
    #fig.savefig("wavelet.png")
    fig.show()
    input("Press Enter to continue...")


# ---------------------------ooo0ooo---------------------------
def CalcRunningMeanPower(tr, deltaT):
    N = len(tr)
    dt = tr.stats.delta
    newStream = tr.copy()
    x = newStream.data

    x = x ** 2

    nSamplePoints = int(deltaT / dt)
    runningMean = np.zeros((N - nSamplePoints), np.float32)

    # determie first tranche
    tempSum = 0.0

    for i in range(0, (nSamplePoints - 1)):
        tempSum = tempSum + x[i]

        runningMean[i] = tempSum

    # calc rest of the sums by subracting first value and adding new one from far end
    for i in range(1, (N - (nSamplePoints + 1))):
        tempSum = tempSum - x[i - 1] + x[i + nSamplePoints]
        runningMean[i] = tempSum
    # calc averaged acoustic intensity as P^2/(density*c)
    density_times_c = (1.2 * 330)
    runningMean = runningMean / (density_times_c)

    newStream.data = runningMean
    newStream.stats.npts = len(runningMean)

    return newStream


# ---------------------------ooo0ooo---------------------------
def DayPlotAcousticPower(tr, deltaT):
    st2 = CalcRunningMeanPower(tr, deltaT)
    st2.plot(type="dayplot", title='test', data_unit='$Wm^{-2}$', interval=60, right_vertical_labels=False,
             one_tick_per_line=False, color=['k', 'r', 'b', 'g'], show_y_UTC_label=False)


# ---------------------------ooo0ooo---------------------------
def PlotAcousticPower(tr):
    st2 = CalcRunningMeanPower(tr)
    st2.plot(title='test', data_unit='$Wm^{-2}$', show_y_UTC_label=False)


# ---------------------------ooo0ooo---------------------------
def plotMany():
    st1 = opendataFile()
    st2 = opendataFile()
    st3 = opendataFile()

    # st.plot()
    st1.detrend(type='demean')
    st2.detrend(type='demean')
    st3.detrend(type='demean')


    tr1 = st1[0].copy()
    tr2 = st2[0].copy()
    tr3 = st3[0].copy()


    lowCut = 1.0
    highCut = 15.0


    tr1x=tr1.filter('bandpass', freqmin=lowCut, freqmax=highCut, corners=4, zerophase=True)
    tr2x=tr2.filter('bandpass', freqmin=lowCut, freqmax=highCut, corners=4, zerophase=True)
    tr3x=tr3.filter('bandpass', freqmin=lowCut, freqmax=highCut, corners=4, zerophase=True)
    startMinute = 780
    endMinute = 900
    tracestart = tr1.stats.starttime
    startSec = (startMinute * 60)
    endSec = (endMinute * 60)
    tr1x.trim(tracestart + startSec, tracestart + endSec)
    tr2x.trim(tracestart + startSec, tracestart + endSec)
    tr3x.trim(tracestart + startSec, tracestart + endSec)
    
    # Here is where you can apply filters
    # 
    # 
    
    #deltaT = 5.0
    #tr1 = CalcRunningMeanPower(tr1, deltaT)
    #tr2 = CalcRunningMeanPower(tr2, deltaT)
    #tr3 = CalcRunningMeanPower(tr3, deltaT)


    plotMultiple(tr1x, tr2x, tr3x,  lowCut, highCut)


# ---------------------------ooo0ooo---------------------------
def plotMultiple(tr1, tr2, lowCut, highCut):
    # This is a general routine to simply plot 3 streams without 
    # processing which is expected to be performed in the calling
    # routine
    
    N1 = len(tr1.data)
    N2 = len(tr2.data)
    #N3 = len(tr3.data)

    samplingFreq = (tr1.stats.sampling_rate + tr2.stats.sampling_rate) / 2.0

    x1 = np.linspace(1, (N1 / samplingFreq), N1)
    x1 = np.divide(x1, 3600)

    x2 = np.linspace(1, (N2 / samplingFreq), N2)
    x2 = np.divide(x2, 3600)

    # x3 = np.linspace(1, (N3 / samplingFreq), N3)
    # x3 = np.divide(x3, 3600)


    TitleString = str(lowCut) + '-' + str(highCut) + ' Hz'
    fig = plt.figure(figsize=(14, 9))
    fig.canvas.set_window_title(TitleString)
    fig.suptitle('Filtered ' + str(lowCut) + '--' + str(highCut) + 'Hz')

    plt.subplots_adjust(hspace=0.001)
    gs = gridspec.GridSpec(2, 1)

    ax1 = plt.subplot(gs[0])
    ax1.plot(x1, tr1, color='k')
   # ax1.set_xlim([0, 24])
    ax1.legend([str(tr1.stats.starttime.date)], loc='upper right', fontsize=12)

    ax2 = plt.subplot(gs[1])
    ax2.plot(x2, tr2, color='k')
    #ax2.set_xlim([0, 24])
    ax2.legend([str(tr2.stats.starttime.date)], loc='upper right', fontsize=12)
    #ax2.axvline(x=7.4, color='r')

    # ax3 = plt.subplot(gs[2])
    # ax3.plot(x3, tr3, color='k')
    # #ax3.set_xlim([0, 24])
    # ax3.legend([str(tr3.stats.starttime.date)], loc='upper right', fontsize=12)
    # #ax3.axvline(x=17, color='b')


    fig.tight_layout()
    fig.savefig("test.png")
    plt.show()
    print('press return to contine')
    return


# ---------------------------ooo0ooo---------------------------

# ---------------------------ooo0ooo---------------------------
def plotDayplot(tr):
    tr.plot(type="dayplot", title='my title', data_unit='$\Delta$Pa', interval=60, right_vertical_labels=False,
            one_tick_per_line=False, color=['k', 'r', 'b', 'g'], show_y_UTC_label=False)
    return


# ---------------------------ooo0ooo---------------------------
def simplePlot(tr):
    tr.plot()
    return


# ---------------------------ooo0ooo---------------------------
def plotpowermagnitudeSpectrum(tr):
    print('plotting magnitude spectrum....')
    ppsd = PPSD(tr.stats, metadata=' ')
    ppsd.add(tr)
    ppsd.plot()
    return


# ---------------------------ooo0ooo---------------------------
def plotsSimpleFFT(tr):
    print('plotting FFT....')
    print(tr.stats)

    dt = tr.stats.delta
    Fs = 1 / dt  # sampling frequency
    tracestart = tr.stats.starttime

    t = np.arange(startSec, endSec, dt)  # create np array for time axis
    sigTemp = tr2.data
    s = sigTemp[0:len(t)]

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(7, 7))

    # plot time signal:
    axes[0, 0].set_title("Signal")
    axes[0, 0].plot(t, s, color='C0')
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Amplitude")

    # plot different spectrum types:
    axes[1, 0].set_title("Magnitude Spectrum")
    axes[1, 0].magnitude_spectrum(s, Fs=Fs, color='C1')

    axes[1, 1].set_title("Log. Magnitude Spectrum")
    axes[1, 1].magnitude_spectrum(s, Fs=Fs, scale='dB', color='C1')

    axes[2, 0].set_title("Phase Spectrum ")
    axes[2, 0].phase_spectrum(s, Fs=Fs, color='C2')

    axes[2, 1].set_title("Power Spectrum Density")
    axes[2, 1].psd(s, 256, Fs, Fc=1)

    axes[0, 1].remove()  # don't display empty ax

    fig.tight_layout()
    plt.show()
    return


# ---------------------------ooo0ooo---------------------------
def plotmagnitudeSpectrum(tr):
    print('plotting magnitude spectrum....')

    dt = tr.stats.delta
    Fs = 1 / dt  # sampling frequency
    tracestart = tr.stats.starttime

    t = np.arange(startSec, endSec, dt)  # create np array for time axis
    sigTemp = tr.data
    s = sigTemp[0:len(t)]
    st1 = opendataFile()  # select a data file to work on
    tr1 = st1[0].copy()  # always work with a copy
    tr1.detrend(type='demean') # effectivly zeroes the trace on the mean value

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(7, 7))

    # plot time signal:
    axes[0].set_title("Signal")
    axes[0].plot(t, s, color='C0')
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Amplitude")

    # plot spectrum types:
    axes[1].set_title("Magnitude Spectrum")
    axes[1].magnitude_spectrum(s, Fs=Fs, color='C1')

    axes[2].set_title("Log. Magnitude Spectrum")
    axes[2].magnitude_spectrum(s, Fs=Fs, scale='dB', color='C1')

    fig.tight_layout()
    plt.show()
    return


# ---------------------------ooo0ooo---------------------------
def plotmagnitudeSpectrum2(tr):
    print('plotting magnitude spectrum....')

    dt = tr.stats.delta
    Fs = 1 / dt  # sampling frequency
    tracestart = tr.stats.starttime

    t = np.arange(startSec, endSec, dt)  # create np array for time axis
    sigTemp = tr2.data
    s = sigTemp[0:len(t)]

    fig, ax = plt.subplots()

    # plot spectrum types:
    ax.set_title("Magnitude Spectrum")
    ax.grid()
    ax.magnitude_spectrum(s, Fs=Fs, color='C1')

    fig.savefig("test.png")

    plt.show()
    return


# ---------------------------ooo0ooo---------------------------
def opendataFile():
    root = tk.Tk()
    root.withdraw()
    root.filename = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select file",
                                               filetypes=[("miniseed data files", "*.mseed")])
    st = read(root.filename)
    return st
# ---------------------------ooo0ooo---------------------------

def openFolder():
	root = Tk()
	root.withdraw()
	#folder_selected = filedialog.askdirectory()
	folder_selected = filedialog.askdirectory(initialdir=os.getcwd(), title="Select directory")
	return folder_selected

# ---------------------------ooo0ooo---------------------------
def readInFolder():
	z=openFolder()
	st = Stream()
	# Read in all files within slected directory.
	listing = os.listdir(z)
	for file in listing:
		if '.mseed' in file:
			print(file)
			streamTemp=obspy.read(z+'/'+file)
			#need to resample at below the lowest sample rate to concatendate
			#hourly datafiles. This code was written for 40Hz sampling
			# however varies slightly between saves so resampling required
			streamTemp.resample(30.0000) 
			st += streamTemp
	st.sort(['starttime'])
	st.merge(method=1, fill_value=0.00)
	return st

# ---------------------------ooo0ooo---------------------------


# ---------------------------------#
#                                  #
#             Main Body            #
#                                  #
# ---------------------------------#

deltaT = 5.0  	# time interval seconds to calculate running mean for acoustic power
lowCut = 0.01 	# low frequency cut-off
highCut = 10.0# high frequency cut-off


###~~~read in single datafile

# st1 = opendataFile()  # select a data file to work on
# tr1 = st1[0].copy()  # always work with a copy
# print(tr1.stats)
# tr1.detrend(type='demean') # effectivly zeroes the trace on the mean value
# tr1.plot()








### -=- read in a days worth of data
st = readInFolder()
tr=st[0].copy()# always work with a copy
tr.filter('bandpass', freqmin=lowCut, freqmax=highCut, corners=4, zerophase=True)
#tr.plot()
#DayPlotAcousticPower
plotDayplot(tr)











#st.plot()
tr=st[0].copy()# always work with a copy
tr.filter('bandpass', freqmin=lowCut, freqmax=highCut, corners=4, zerophase=True)
#tr.plot()
#DayPlotAcousticPower(tr, deltaT)
plotDayplot(tr)
# ### ----- select slice of data to work on
# startMinute = 1140
# endMinute =  1200
# tracestart = tr1.stats.starttime
# startSec = (startMinute * 60)
# endSec = (endMinute * 60)
# tr1.trim(tracestart + startSec, tracestart + endSec)
# #tr1.plot()
# #plotRawSignalBands(tr1)
# tr1.plot()




#
#tr1.resample(30.0000) 
#tr1.filter('bandpass', freqmin=lowCut, freqmax=highCut, corners=4, zerophase=True)
#print(len(tr1.data))
#tr1.plot

# st2 = opendataFile()  # select a data file to work on
# tr2 = st2[0].copy()  # always work with a copy
# tr2.detrend(type='demean') # effectivly zeroes the trace on the mean value
# tr2.data = tr2.data*-1.0
# tr2.resample(30.0000) 
# tr2.filter('bandpass', freqmin=lowCut, freqmax=highCut, corners=4, zerophase=True)
# print(len(tr2.data))

# ---------------------------------#
# high & low pass filter
#tr1.filter('bandpass', freqmin=lowCut, freqmax=highCut, corners=4, zerophase=True)




# #plotMultiple(tr1, tr2, tr3, lowCut, highCut)

#plotRawSignalBands(tr1)


# ### ----- select slice of data to work on
# startMinute = 1380
# endMinute =  1439

# tracestart = tr1.stats.starttime
# startSec = (startMinute * 60)
# endSec = (endMinute * 60)
# tr1.trim(tracestart + startSec, tracestart + endSec)

# tracestart = tr2.stats.starttime
# startSec = (startMinute * 60)
# endSec = (endMinute * 60)
# tr2.trim(tracestart + startSec, tracestart + endSec)

#plotMultiple(tr1, tr2, lowCut, highCut)
#plotRawSignalBands(tr1)

# plotWaveletTransform(tr1, lowCut, highCut)

# plotWaveletTransform(tr2, lowCut, highCut)


# st3 = opendataFile()  # select a data file to work on
# tr3 = st3[0].copy()  # always work with a copy
# tr3.detrend(type='demean') # effectivly zeroes the trace on the mean value
# ### ----- select slice of data to work on
# startMinute = 60
# endMinute =  120
# tracestart = tr3.stats.starttime
# startSec = (startMinute * 60)
# endSec = (endMinute * 60)
# tr3.trim(tracestart + startSec, tracestart + endSec)






#plotMultiple(tr1, tr2, tr3, lowCut, highCut)
# ---------------------------------#
# simple plot of entire data stream
#tr.plot()






# ---------------------------------#
### ----- select slice of data to work on
# startMinute = 600
# endMinute =  660
# tracestart = tr.stats.starttime
# startSec = (startMinute * 60)
# endSec = (endMinute * 60)
# tr.trim(tracestart + startSec, tracestart + endSec)


# plotWaveletTransform(tr1, lowCut, highCut)
# #tr.plot()
# # # ---------------------------------#
# # # easy 24hr plot    
# DayPlotAcousticPower(tr, deltaT)
# plotDayplot(tr)
# #input("Press Enter to continue...")

# # ---------------------------------#
# # plot raw signal by frequency bands
# plotRawSignalBands(tr1)
# plotAcousticBands(tr, deltaT)
# input("Press Enter to continue...")

# ---------------------------------#
# plot by acoustic power frequency bands
#plotAcousticBands(tr, deltaT)
#input("Press Enter to continue...")





# plotpowermagnitudeSpectrum(tr)

# 
# tr2.plot()

# plotsSimpleFFT(tr)
# plotPeriodogram(tr, lowCut, highCut)
#plotmagnitudeSpectrum(tr)




##plotAcousticBands(tr, deltaT)
##input("Press Enter to continue...")
#plotWaveletTransform(tr2, 0.01, highCut)
#input("Press Enter to continue...")
# st2 = CalcRunningMeanPower(tr2, deltaT)
# st2.plot()
# simplePlot(st2, lowCut, highCut)
# st2.plot(title='test', data_unit='$Wm^{-2}$', show_y_UTC_label=False)





#plotSpectrogram(tr, lowCut, highCut)




# ----------------------------------------------


# ---------------------------ooo0ooo---------------------------
# ---------------------------ooo0ooo---------------------------

