# PySDRVNA Toolkit is a Python toolkit to use your Software Defined Radio
# as a simple Vector Network Analyzer. 
# Copyright (c) 2013 by Steve Haynal, KF7O.

# PySDRVNA Toolkit is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

# PySDRVNA Toolkit is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License at <http://www.gnu.org/licenses/> for details.

from __future__ import print_function
import struct, usb.core, usb.util, traceback, pyfftw, time, math
import numpy as np
import scipy as sp
import jacklib, ctypes, threading, getopt, sys, os
import pickle
import matplotlib as mpl
## Use Qt4Agg, TkAgg or WXAgg as others cause xruns
mpl.use('Qt4Agg',warn=False)
import matplotlib.pyplot as plt

from Measurement import *
import config

#####
## All si570 interface code is borrow directly from or based on QUISK. Thank you!
#####

# All USB access is through control transfers using pyusb.
#   byte_array      = dev.ctrl_transfer (IN,  bmRequest, wValue, wIndex, length, timout)
#   len(string_msg) = dev.ctrl_transfer (OUT, bmRequest, wValue, wIndex, string_msg, timout)
# I2C-address of the SI570;  Thanks to Joachim Schneider, DB6QS
si570_i2c_address = 0x55

# Thanks to Ethan Blanton, KB8OJH, for this patch for the Si570 (many SoftRocks):
# These are used by SetFreqByDirect(); see below.
# The Si570 DCO must be clamped between these values
SI570_MIN_DCO = 4.85e9
SI570_MAX_DCO = 5.67e9
# The Si570 has 6 valid HSDIV values.  Subtract 4 from HSDIV before
# stuffing it.  We want to find the highest HSDIV first, so start
# from 11.
SI570_HSDIV_VALUES = [11, 9, 7, 6, 5, 4]

si570_xtal_freq = 114251870


IN =  usb.util.build_request_type(usb.util.CTRL_IN,  usb.util.CTRL_TYPE_VENDOR, usb.util.CTRL_RECIPIENT_DEVICE)
OUT = usb.util.build_request_type(usb.util.CTRL_OUT, usb.util.CTRL_TYPE_VENDOR, usb.util.CTRL_RECIPIENT_DEVICE)

UBYTE2 = struct.Struct('<H')
UBYTE4 = struct.Struct('<L')    # Thanks to Sivan Toledo

## To supress annoying Jack error messages if script starts jackd
def RedirectStderr():
    sys.stderr.flush()
    newstderr = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)
    sys.stderr = os.fdopen(newstderr, 'w')  


class VNA:
  def __init__(self,configd=None):
    """Create a VNA object"""

    if configd is None: configd = config.__dict__

    self.printlevel = configd['printlevel']
    self.fftn = configd['fftn']
    self.amp = configd['amp']
    self.warmuptime = configd['warmuptime']
    self.cooldowntime = configd['cooldowntime']
   
    self.docapture = threading.Event()
    self.docapture.set()

    self.xrun = threading.Event()
    self.xrun.clear()

    self.startframe = 0

    #self.jackclient = jacklib.client_open("pysdrvna", jacklib.JackNoStartServer | jacklib.JackSessionID, None)
    self.jackclient = jacklib.client_open("pysdrvna", jacklib.JackSessionID, None)
    
    try:
      self.jackclient.contents
    except:
      print("Problems with Jack")
      return
      #raise

    self.iI = jacklib.port_register(self.jackclient,"iI", jacklib.JACK_DEFAULT_AUDIO_TYPE, jacklib.JackPortIsInput, 0)
    self.iQ = jacklib.port_register(self.jackclient,"iQ", jacklib.JACK_DEFAULT_AUDIO_TYPE, jacklib.JackPortIsInput, 0)    

    self.oI = jacklib.port_register(self.jackclient,"oI", jacklib.JACK_DEFAULT_AUDIO_TYPE, jacklib.JackPortIsOutput, 0)
    self.oQ = jacklib.port_register(self.jackclient,"oQ", jacklib.JACK_DEFAULT_AUDIO_TYPE, jacklib.JackPortIsOutput, 0)
   
    jacklib.set_process_callback(self.jackclient, self.JackProcess, 0)
    jacklib.set_xrun_callback(self.jackclient, self.JackXrun, 0)

    jacklib.activate(self.jackclient)
   
    jacklib.connect(self.jackclient,"pysdrvna:oQ", configd['outQ'])
    jacklib.connect(self.jackclient,"pysdrvna:oI", configd['outI'])    
    
    jacklib.connect(self.jackclient,configd['inQ'],"pysdrvna:iQ")
    jacklib.connect(self.jackclient,configd['inI'],"pysdrvna:iI") 
    
    self.Sr = float(jacklib.get_sample_rate(self.jackclient))
    self.dt = 1.0/self.Sr
      
    ## Align frequency to nearest bin
    self.fftbin = int(round((configd['freq']/self.Sr)*self.fftn))      
    self.freq = (float(self.fftbin)/self.fftn) * self.Sr
    
    ## Windowing function
    #self.fftwindow = np.blackman(self.fftn)
    self.fftwindow = np.hanning(self.fftn)
    #self.fftwindow = np.kaiser(self.fftn,14)
    #self.fftwindow = np.hamming(self.fftn)
    #self.fftwindow = np.bartlett(self.fftn)
    #self.fftwindow = None
      
    ## Latency settings
    jlr = jacklib.jack_latency_range_t()
    jacklib.port_get_latency_range(self.oI,jacklib.JackPlaybackLatency,jlr)
    self.minrtframes = jlr.min
    jacklib.port_get_latency_range(self.iI,jacklib.JackCaptureLatency,jlr)
    self.minrtframes += jlr.min
 
    # The above code does not always work
    # Reasonable estimate is 3 times the buffer size
    ## Compute initial array length
    self.buffersz = int(jacklib.get_buffer_size(self.jackclient))
    if self.minrtframes < (3*self.buffersz):
        self.minrtframes = 3 * self.buffersz


    ## rtframes is the round trip audio latency, or when the received audio signal will start    
    self.rtframes = configd['rtframes'] if configd['rtframes'] else self.minrtframes
    ## delta from rtframes to phase shift sync in frames
    self.rtframes2sync = configd['rtframes2sync']
    ## delta from phase shift sync to fft start in frames
    self.sync2fft = configd['sync2fft']
    ## delta from end of fft to end of audio
    self.fft2end = configd['fft2end']


    if configd['rtframes'] is None:
        ## Loose fit if estimating based on minrtframes
        buffers, remainder = divmod((2*self.rtframes) + self.rtframes2sync + self.sync2fft + self.fftn + self.fft2end,self.buffersz)
    else:
        ## Tight fit if rtframes is defined
        buffers, remainder = divmod(self.rtframes + self.rtframes2sync + self.sync2fft + self.fftn + self.fft2end,self.buffersz)
  
    if remainder > 0: buffers = buffers + 1
    self.synci = self.rtframes
    self.InitJackArrays(self.freq,buffers*self.buffersz)
    
    self.fftia = pyfftw.n_byte_align_empty(self.fftn, 16, 'complex128')
    self.fftoa = pyfftw.n_byte_align_empty(self.fftn, 16, 'complex128')
    ## Create FFT Plan
    self.fft = pyfftw.FFTW(self.fftia,self.fftoa)

    self.OpenSoftRock()
    self.Info()

  def InitJackArrays(self,freq,samples):
    """Initialize Jack Arrays"""
    self.iIa = sp.zeros(samples).astype(sp.float32)
    self.iQa = sp.zeros(samples).astype(sp.float32)      
    
    self.oIa = sp.zeros(samples, dtype=sp.float32 )
    self.oQa = sp.zeros(samples, dtype=sp.float32 )
    
    ## 100 frames warmup
    sf = 0
    ef = self.rtframes2sync
    samples = sp.pi + (2*sp.pi*freq*(self.dt * sp.r_[sf:ef]))
    self.oIa[sf:ef] = self.amp * sp.cos(samples)
    self.oQa[sf:ef] = self.amp * sp.sin(samples)
    
    # For IQ balancing
    #self.oIa[sf:ef] = sp.cos(samples) - (sp.sin(samples)*(1+self.oalpha)*sp.sin(self.ophi))
    #self.oQa[sf:ef] = sp.sin(samples)*(1+self.oalpha)*sp.cos(self.ophi)
    
    ## 180 phase change 
    sf = ef
    ef = ef + self.sync2fft + self.fftn + self.fft2end
    samples = (2*sp.pi*freq*(self.dt * sp.r_[sf:ef]))
    self.oIa[sf:ef] = self.amp * sp.cos(samples) 
    self.oQa[sf:ef] = self.amp * sp.sin(samples)   
    
    # For IQ balancing
    #self.oIa[sf:ef] = sp.cos(samples) - (sp.sin(samples)*(1+self.oalpha)*sp.sin(self.ophi))
    #self.oQa[sf:ef] = sp.sin(samples)*(1+self.oalpha)*sp.cos(self.ophi)    
    
  def ResizeArrays(self,rtframes=None):
    """Resize Jack Arrays"""
    ## Estimate new rtframes if None
    if rtframes is None:
      ## Use last sync index
      if self.synci == self.rtframes:
        raise IndexError("Sync index appears uninitialized")
      else:
        rtframes = self.synci - self.rtframes2sync
        print("RTFrames computed from last Sync index",rtframes)
    
    ## Tight fit
    buffers, remainder = divmod(rtframes + self.rtframes2sync + self.sync2fft + self.fftn + self.fft2end,self.buffersz)
    if remainder > 0: buffers = buffers + 1
    
    
    print("Array length was",self.iIa.size,end=" ")
    if buffers*self.buffersz != self.iIa.size:
      self.InitJackArrays(self.freq,buffers*self.buffersz)
    print("now",self.iIa.size)
    
    print("RTFrames was",self.rtframes,"now",rtframes)
    self.rtframes = rtframes
    
  def CalibrateArrays(self):
    """Calibrate Array Lengths Assuming Good Audio Levels"""
    i = 5
    while i > 0:
      try:
        self.Test()
        self.ResizeArrays()
        i = 0
      except:
        self.ResizeArrays(2*self.rtframes)
        i -= 1

  def NewAmp(self,amp):
    """Regenerate Test Tone with New Amplitude"""
    print("Amplitude was",self.amp,"now",amp)
    if self.amp != amp:
      self.amp = amp
      self.InitJackArrays(self.freq,self.iIa.size)

  def Info(self):
    """Print Information"""
    print("FFT Size:",self.fftn,"FFT Bin:",self.fftbin,"Test Freq:",self.freq,"Amp:",self.amp,"RT Frames:",self.rtframes,end=" ")
    print("RT2Sync:",self.rtframes2sync,"Sync2FFT:",self.sync2fft,"FFT2End:",self.fft2end,"Array Length:",self.iIa.size,"Sync Index:",self.synci)
    
    
  ## SoftRock Control
  def OpenSoftRock(self):
    """Open the SoftRock"""
    ## Open SoftRock
    # find our device
    self.usb_vendor_id = 0x16c0
    self.usb_product_id = 0x05dc
    self.usb_dev = usb.core.find(idVendor=0x16c0, idProduct=0x05dc)
    if self.usb_dev is None:
      print('USB device not found VendorID 0x%X ProductID 0x%X' % (self.usb_vendor_id, self.usb_product_id))
    else:
      try:
        self.usb_dev.set_configuration()
        ret = self.usb_dev.ctrl_transfer(IN, 0x00, 0x0E00, 0, 2)
        if len(ret) == 2:
          ver = "%d.%d" % (ret[1], ret[0])
        else:
          ver = 'unknown'
        print('Capture from SoftRock Firmware %s' % ver)
        print('Startup freq', self.GetStartupFreq())
        print('Run freq', self.GetFreq())
        print('Address 0x%X' % self.usb_dev.ctrl_transfer(IN, 0x41, 0, 0, 1)[0])
        sm = self.usb_dev.ctrl_transfer(IN, 0x3B, 0, 0, 2)
        sm = UBYTE2.unpack(sm)[0]
        print('Smooth tune', sm)
      except:
        print("No permission to access the SoftRock USB interface")
        self.usb_dev = None

  def GetStartupFreq(self): # return the startup frequency / 4
    """Return the SoftRock Startup Frequency"""
    if not self.usb_dev: return 0
    ret = self.usb_dev.ctrl_transfer(IN, 0x3C, 0, 0, 4)
    s = ret.tostring()
    freq = UBYTE4.unpack(s)[0]
    freq = int(freq * 1.0e6 / 2097152.0 / 4.0 + 0.5)
    return freq
    
  def GetFreq(self):    # return the running frequency / 4
    """Return the SoftRock Running Frequency"""
    if not self.usb_dev: return 0
    ret = self.usb_dev.ctrl_transfer(IN, 0x3A, 0, 0, 4)
    s = ret.tostring()
    freq = UBYTE4.unpack(s)[0]
    freq = int(freq * 1.0e6 / 2097152.0 / 4.0 + 0.5)
    return freq    
    
  def SetFreq(self, freq):
    """Set the SoftRock Running Frequency"""
    freq = int(freq/1.0e6 * 2097152.0 * 4.0 + 0.5)
    s = UBYTE4.pack(freq)
    try:
      self.usb_dev.ctrl_transfer(OUT, 0x32, si570_i2c_address + 0x700, 0, s)
    except usb.core.USBError:
      traceback.print_exc()
    
  def SetFreqNew(self, freq): # Thanks to Ethan Blanton, KB8OJH
    if freq == 0.0:
      return False
    # For now, find the minimum DCO speed that will give us the
    # desired frequency; if we're slewing in the future, we want this
    # to additionally yield an RFREQ ~= 512.
    freq = int(freq * 4)
    dco_new = None
    hsdiv_new = 0
    n1_new = 0
    for hsdiv in SI570_HSDIV_VALUES:
      n1 = int(math.ceil(SI570_MIN_DCO / (freq * hsdiv)))
      if n1 < 1:
        n1 = 1
      else:
        n1 = ((n1 + 1) / 2) * 2
      dco = (freq * 1.0) * hsdiv * n1
      # Since we're starting with max hsdiv, this can only happen if
      # freq was larger than we can handle
      if n1 > 128:
        continue
      if dco < SI570_MIN_DCO or dco > SI570_MAX_DCO:
        # This really shouldn't happen
        continue
      if not dco_new or dco < dco_new:
        dco_new = dco
        hsdiv_new = hsdiv
        n1_new = n1
    if not dco_new:
      # For some reason, we were unable to calculate a frequency.
      # Probably because the frequency requested is outside the range
      # of our device.
      return False    # Failure
    rfreq = dco_new / si570_xtal_freq
    rfreq_int = int(rfreq)
    rfreq_frac = int(round((rfreq - rfreq_int) * 2**28))
    # It looks like the DG8SAQ protocol just passes r7-r12 straight
    # To the Si570 when given command 0x30.  Easy enough.
    # n1 is stuffed as n1 - 1, hsdiv is stuffed as hsdiv - 4.
    hsdiv_new = hsdiv_new - 4
    n1_new = int(n1_new - 1)
    print(hsdiv_new,n1_new,rfreq_int,rfreq_frac)
    s = struct.Struct('>BBL').pack((hsdiv_new << 5) + (n1_new >> 2),
                                   ((n1_new & 0x3) << 6) + (rfreq_int >> 4),
                                   ((rfreq_int & 0xf) << 28) + rfreq_frac)
    self.usb_dev.ctrl_transfer(OUT, 0x30, si570_i2c_address + 0x700, 0, s)
    return True   # Success


  def PTT(self, ptt):
    if self.usb_dev:
      try:
        self.usb_dev.ctrl_transfer(IN, 0x50, ptt, 0, 3)
      except usb.core.USBError:
        traceback.print_exc()
    

  def Exit(self):
    """Exit Cleanly from Interactive VNA"""
    try:
      jacklib.deactivate(self.jackclient)
    except:
      pass
    try:
      jacklib.client_close(self.jackclient)
    except:
      pass
    exit()
    

  def Sync(self):
    """Locate the Sync Phase Shift"""
    ## Find start by amplitude
    mv = 0.7 * self.iIa[self.minrtframes:].max()
    sia = np.nonzero( self.iIa[self.minrtframes:] > mv )[0]
    si = sia[0] + self.minrtframes
    
    ## Phase change
    synca = self.iIa[si:si+(2*self.rtframes2sync)] - 1j * self.iQa[si:si+(2*self.rtframes2sync)]    
    anglea = np.angle(synca,deg=True) + 180
    deltaaa = (anglea[1:] - anglea[:-1]) % 360
    
    ## Find sync index
    try:
        syncindex = (np.nonzero( (deltaaa > 90) & (deltaaa < 270) )[0][0]) + si 
    except:
        raise IndexError("Jack arrays are not long enough and/or bad sync. Resize arrays.")
        
    self.synci = syncindex
    
    if self.iIa.size < (self.synci + self.sync2fft + self.fftn):
      raise IndexError("Jack arrays are not long enough and/or bad sync. Resize arrays.")
    
    
  def DoFFT(self):
    """Calculate the FFT"""
    I = self.iIa[self.synci+self.sync2fft:self.synci+self.sync2fft+self.fftn]
    Q = self.iQa[self.synci+self.sync2fft:self.synci+self.sync2fft+self.fftn]

    ## Remove DC bias
    #Q = Q - np.mean(Q)
    #I = I - np.mean(I)
    
    self.fftia[:] = I - 1j * Q
    if self.fftwindow != None: self.fftia[:] = self.fftwindow * self.fftia
    
    self.fft()

  def DoCoolDown(self):
    if self.cooldowntime:
      time.sleep(self.cooldowntime)

  def DoWarmUp(self):
    if self.warmuptime:
      self.PTT(1)
      time.sleep(self.warmuptime)
      self.PTT(0)
      self.DoCoolDown()

  def Test(self,iterations=1):
    """Execute a Test Measurement for Specified Iterations"""
    self.DoWarmUp()

    for i in range(0,iterations):
      print(i,end=" ")
      self.M()
      self.Mprint()

      self.DoCoolDown()

  
  def JackProcess(self,nframes,arg):
    """Main Jack Process Method"""
    if not self.docapture.is_set():
      
      ## Copy input data
      endframe = self.startframe + nframes
      if endframe > self.iIa.size:
        endframe = self.iIa.size
        
      tsz = (endframe-self.startframe) * ctypes.sizeof(jacklib.jack_default_audio_sample_t)
      basei = self.startframe * ctypes.sizeof(jacklib.jack_default_audio_sample_t)
    
      ctypes.memmove(self.iIa.ctypes.data+basei,jacklib.port_get_buffer(self.iI,nframes),tsz)
      ctypes.memmove(self.iQa.ctypes.data+basei,jacklib.port_get_buffer(self.iQ,nframes),tsz) 
      ctypes.memmove(jacklib.port_get_buffer(self.oI,nframes),self.oIa.ctypes.data+basei,tsz)
      ctypes.memmove(jacklib.port_get_buffer(self.oQ,nframes),self.oQa.ctypes.data+basei,tsz)
      #print self.startframe,endframe,basei,tsz

      self.startframe = endframe
      if self.startframe >= self.iIa.size:
        self.docapture.set() 
              
    return 0

  def JackXrun(self,arg):
    """Jack Xrun Callback"""
    self.xrun.set()
    return 0

    
  def Mprint(self,isdut=False):
    """Print Information for a Measurement"""
    if self.printlevel > 0:
      print("Sync:%d" % self.synci,end=" ")
      print("Freq:%d" % int(round(self.GetFreq()+self.freq)),end=" ")
      cn = self.fftoa[self.fftbin]
      print("Real:%3.2f" % cn.real,end=" ")
      print("Imag:%3.2f" % cn.imag,end=" ")
      print("Mag:%3.2f" % np.abs(cn),end=" ")
      print("Phase:%3.2f" % np.angle(cn,deg=True))
      
      #print "Bins",np.abs(self.fftoa[self.fftbin-1]),np.abs(self.fftoa[self.fftbin]),np.abs(self.fftoa[self.fftbin+1])
      
  def M(self,freq=None,ptt=True,warmup=False):
    """Main Measurement Method"""
    if freq: 
      #time.sleep(0.2)
      self.SetFreq(freq-self.freq)
      time.sleep(0.005)

    if warmup: self.DoWarmUp()
    
    attempts = 5
    while attempts > 0:
      try:
        if ptt: self.PTT(1) 
    
        self.startframe = 0
        self.xrun.clear()
        self.docapture.clear()
        self.docapture.wait()

        if ptt: self.PTT(0)

        if self.xrun.is_set():
          ## An xrun occurred, attempt again
          print("XRUN during measurement, retrying")
          attempts = attempts -1
        else:
          self.Sync()
          attempts = 0
      except:
        print("Error during measurement, retrying")
        attempts = attempts - 1
      
    self.DoFFT()
    
  def M2Array(self,freq,array):
    """Array of Main Measurements Method"""
    try:

      self.DoWarmUp()
    
      i = 0
      for f in freq:
        self.M(int(f*1000000))
        array[i] = self.fftoa[self.fftbin]
        self.Mprint()
        i += 1
        self.DoCoolDown()
    except:
      print("Error during array measurement")
    self.PTT(0)    

  def MO(self,m):
    """Measure Open Standard"""
    print("Beginning Open Measurements")
    self.M2Array(m.freq,m.open)
    
  def MS(self,m):
    """Measure Short Standard"""
    print("Beginning Short Measurements")
    self.M2Array(m.freq,m.short)
    
  def ML(self,m):
    """Measure Load Standard"""
    print("Beginning Load Measurements")
    self.M2Array(m.freq,m.load)

  def MD(self,m):
    """Measure DUT"""
    print("Beginning DUT Measurements")
    self.M2Array(m.freq,m.dut)
    
  def SWR(self,m,iterations=100):
    """Make Iteration Number of SWR Measurements at Current Frequency"""
    if m.freq.size != 1:
      raise IndexError("SWR requires exactly 1 measurement")
    print("Beginning SWR Measurements")

    self.DoWarmUp()

    for j in range(0,iterations):
      self.M(int(m.freq[0]*1000000))
      m.dut[0] = self.fftoa[self.fftbin]
      m.PrintSWR()
      self.DoCoolDown()
 
  def PlotTD(self):
    """Plot Time Domain of Jack Input and Output Arrays for Last Measurement"""
    fig = self.CreateFigure("Time Domain")
    sp = fig.add_subplot(111)
    xaxis = range(0,len(self.iIa))
    sp.plot(xaxis,self.iIa,'.-',color='b',label='iI')
    ## 180 phase shift as complex portion is created with -1j
    sp.plot(xaxis,-1*self.iQa,'.-',color='r',label='iQ')
    sp.plot(xaxis,self.oIa,'.-',color='c',label='oI')
    sp.plot(xaxis,self.oQa,'.-',color='m',label='oQ') 
    ## Identify RTFrames
    maxy = self.oIa.max()
    sp.plot([self.rtframes,self.rtframes],[-maxy,maxy],'k-',lw=3,label='RT Frames')
    ## Identify Sync Index
    sp.plot([self.synci+self.sync2fft,self.synci+self.sync2fft],[-maxy,maxy],'g-',lw=3,label='FFT Start')
    sp.plot([self.synci+self.sync2fft+self.fftn,self.synci+self.sync2fft+self.fftn],[-maxy,maxy],'y-',lw=3,label='FFT End')      
    sp.set_ylabel("Magnitude")
    sp.set_xlabel("Sample")
    #sp.legend(bbox_to_anchor=(1,-0.1))  
    sp.legend(loc=2,bbox_to_anchor=(0,-0.1),ncol=7)   
    plt.show()        
    
 
  
  def PlotFFTInput(self):
    """Plot Time Domain of FFT Input Slice for Last Measurement"""
    fig = self.CreateFigure("FFT Input")
    sp = fig.add_subplot(111)
    xaxis = range(0,self.fftn)
    sp.plot(xaxis,[x.real for x in self.fftia],'.-',color='b',label='I')
    sp.plot(xaxis,[x.imag for x in self.fftia],'.-',color='r',label='Q')    
    sp.set_ylabel("Magnitude")
    sp.set_xlabel("Sample")
    #sp.legend(bbox_to_anchor=(1,-0.1))  
    sp.legend(loc=2,bbox_to_anchor=(0,-0.1),ncol=4)   
    plt.show()    
    

  def PlotFD(self,dbfs=True):
    """Plot Frequency Domain for Last Measurement"""
    freqspectrum = np.abs(self.fftoa)
    freqspectrum = np.concatenate( [freqspectrum[self.fftn/2:self.fftn],freqspectrum[0:self.fftn/2]] )
    if dbfs:
      zerodb = 20*np.log10(self.fftn/2)
      freqspectrum = (20*np.log10(abs(freqspectrum))) - zerodb
      
    fig = self.CreateFigure("Frequency Domain")
    sp = fig.add_subplot(111)
    
    xaxis = np.r_[0:self.fftn] * (self.Sr/self.fftn)
    xaxis = np.concatenate( [(xaxis[self.fftn/2:self.fftn] - self.Sr),xaxis[0:self.fftn/2]])
    sp.plot(xaxis,freqspectrum,'.-',color='b',label='Spectrum')
    sp.set_ylabel("dBFS")
    sp.set_xlabel("Frequency")
    sp.legend(loc=2,bbox_to_anchor=(0,-0.1),ncol=4)   
    plt.show()          
      
    
  def CreateFigure(self,title):
    fig = plt.figure()
    fig.subplots_adjust(bottom=0.2)
    fig.suptitle(title, fontsize=20)
    return fig    


  def UncorrectedReflectionCoefficientAccuracyTest(self,repititions=200,freq=None):
    '''Measure accuracy of uncorrect reflection coefficient for large number of repititions'''
    if freq:
      self.SetFreq(freq)
      time.sleep(0.005)   

    ## Create NP array with proper dimensions
    a = np.zeros( repititions, dtype=np.complex )

    self.DoWarmUp()

    for i in range(repititions):
      self.M()
      cn = self.fftoa[self.fftbin]
      a[i] = cn

      self.DoCoolDown()
     
    pa = np.angle(a)
    magstd = np.std(a)
    angstd = np.std(pa)
    mean = np.mean(a)
    nmagstd = magstd / np.abs(mean)
    print ("STD Deviation of magnitude (percent) and angle:",nmagstd,angstd)
    print ("Mean:",mean,"Range:",np.max(a)-np.min(a))

    return a



if __name__ == '__main__':

  try:
    __IPYTHON__
  except:
    RedirectStderr()
  vna = VNA(config.__dict__)
  print('Usage: python3 VNA.py')
  print("vna is the VNA object. Type help(vna) for more information.")
  print("Use vna.Exit() to exit cleanly. Edit config.py for options.")
  
