#!/usr/bin/python -i

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



import scipy, pylab, struct, usb.core, usb.util, traceback, pyfftw, numpy, time
import jacklib, jacklib_helpers, ctypes, threading
import cPickle as pickle
import matplotlib as mpl
import matplotlib.pyplot as plt

from Measurement import *

#####
## All si570 interface code is borrow directly from or based on QUISK. Thank you!
#####

# All USB access is through control transfers using pyusb.
#   byte_array      = dev.ctrl_transfer (IN,  bmRequest, wValue, wIndex, length, timout)
#   len(string_msg) = dev.ctrl_transfer (OUT, bmRequest, wValue, wIndex, string_msg, timout)
# I2C-address of the SI570;  Thanks to Joachim Schneider, DB6QS
si570_i2c_address = 0x55

IN =  usb.util.build_request_type(usb.util.CTRL_IN,  usb.util.CTRL_TYPE_VENDOR, usb.util.CTRL_RECIPIENT_DEVICE)
OUT = usb.util.build_request_type(usb.util.CTRL_OUT, usb.util.CTRL_TYPE_VENDOR, usb.util.CTRL_RECIPIENT_DEVICE)

UBYTE2 = struct.Struct('<H')
UBYTE4 = struct.Struct('<L')	# Thanks to Sivan Toledo


class VNA:
  def __init__(self,fftn=1024,freq=2350.0,amp=1.0,printlevel=1):
    
    self.printlevel = printlevel
    self.amp = amp
   
    self.docapture = threading.Event()
    self.docapture.set()
    self.startframe = 0
    
    self.jackclient = jacklib.client_open("pysdrvna", jacklib.JackNoStartServer | jacklib.JackSessionID, None)

    self.iI = jacklib.port_register(self.jackclient,"iI", jacklib.JACK_DEFAULT_AUDIO_TYPE, jacklib.JackPortIsInput, 0)
    self.iQ = jacklib.port_register(self.jackclient,"iQ", jacklib.JACK_DEFAULT_AUDIO_TYPE, jacklib.JackPortIsInput, 0)    

    self.oI = jacklib.port_register(self.jackclient,"oI", jacklib.JACK_DEFAULT_AUDIO_TYPE, jacklib.JackPortIsOutput, 0)
    self.oQ = jacklib.port_register(self.jackclient,"oQ", jacklib.JACK_DEFAULT_AUDIO_TYPE, jacklib.JackPortIsOutput, 0)
   
    jacklib.set_process_callback(self.jackclient, self.JackProcess, 0)

    jacklib.activate(self.jackclient)
   
    jacklib.connect(self.jackclient,"pysdrvna:oQ", "system:playback_1")
    jacklib.connect(self.jackclient,"pysdrvna:oI", "system:playback_2")    
    
    jacklib.connect(self.jackclient,"system:capture_1","pysdrvna:iQ")
    jacklib.connect(self.jackclient,"system:capture_2","pysdrvna:iI") 
    
    self.Sr = float(jacklib.get_sample_rate(self.jackclient))
    self.dt = 1.0/self.Sr
        
    self.fftn = fftn
    ## Set FFT bin and frequency to be around 2.3kHz
    self.fftbin = int(round((freq/self.Sr)*self.fftn))      
    self.freq = (float(self.fftbin)/self.fftn) * self.Sr
    
    
    ## Windowing function
    #self.fftwindow = numpy.blackman(self.fftn)
    self.fftwindow = None
      
    ## Latency settings
    jlr = jacklib.jack_latency_range_t()
    jacklib.port_get_latency_range(self.oI,jacklib.JackPlaybackLatency,jlr)
    self.rtframes = jlr.min
    jacklib.port_get_latency_range(self.iI,jacklib.JackCaptureLatency,jlr)
    self.rtframes += jlr.min
    
    ## Compute initial array length
    buffersz = int(jacklib.get_buffer_size(self.jackclient))
    buffers, remainder = divmod(self.rtframes + self.fftn + 256,buffersz)
    if remainder > 0: buffers = buffers + 1

    self.synci = self.rtframes
     
    self.InitJackArrays(self.freq,buffers*buffersz)
    
    self.fftia = pyfftw.n_byte_align_empty(self.fftn, 16, 'complex128')
    self.fftoa = pyfftw.n_byte_align_empty(self.fftn, 16, 'complex128')
    ## Create FFT Plan
    self.fft = pyfftw.FFTW(self.fftia,self.fftoa)
    

  def InitJackArrays(self,freq,samples):

    self.iIa = scipy.zeros(samples).astype(scipy.float32)
    self.iQa = scipy.zeros(samples).astype(scipy.float32)      
    
    self.oIa = scipy.zeros(samples, dtype=scipy.float32 )
    self.oQa = scipy.zeros(samples, dtype=scipy.float32 )
    
    ## 100 frames warmup
    sf = 0
    ef = 100
    samples = scipy.pi + (2*scipy.pi*freq*(self.dt * scipy.r_[sf:ef]))
    self.oIa[sf:ef] = self.amp * scipy.cos(samples)
    self.oQa[sf:ef] = self.amp * scipy.sin(samples)
    
    # For IQ balancing
    #self.oIa[sf:ef] = scipy.cos(samples) - (scipy.sin(samples)*(1+self.oalpha)*scipy.sin(self.ophi))
    #self.oQa[sf:ef] = scipy.sin(samples)*(1+self.oalpha)*scipy.cos(self.ophi)
    
    ## 180 phase change then fftn+50 frames
    sf = ef
    ef = ef + self.fftn + 50
    samples = (2*scipy.pi*freq*(self.dt * scipy.r_[sf:ef]))
    self.oIa[sf:ef] = self.amp * scipy.cos(samples) 
    self.oQa[sf:ef] = self.amp * scipy.sin(samples)   
    
    # For IQ balancing
    #self.oIa[sf:ef] = scipy.cos(samples) - (scipy.sin(samples)*(1+self.oalpha)*scipy.sin(self.ophi))
    #self.oQa[sf:ef] = scipy.sin(samples)*(1+self.oalpha)*scipy.cos(self.ophi)    
    
      
  def Info(self):
    print "FFT Size:",self.fftn,"FFT Bin:",self.fftbin,"Test Freq:",self.freq
    print "Array Length:",self.iIa.size,"Sync Index",self.synci   
    
    
  ## SoftRock Control
  def OpenSoftRock(self):
    ## Open SoftRock
    # find our device
    self.usb_vendor_id = 0x16c0
    self.usb_product_id = 0x05dc
    self.usb_dev = usb.core.find(idVendor=0x16c0, idProduct=0x05dc)
    if self.usb_dev is None:
      print 'USB device not found VendorID 0x%X ProductID 0x%X' % (self.usb_vendor_id, self.usb_product_id)
    else:
      try:
        self.usb_dev.set_configuration()
        ret = self.usb_dev.ctrl_transfer(IN, 0x00, 0x0E00, 0, 2)
        if len(ret) == 2:
          ver = "%d.%d" % (ret[1], ret[0])
        else:
          ver = 'unknown'
        print 'Capture from SoftRock Firmware %s' % ver
        print 'Startup freq', self.GetStartupFreq()
        print 'Run freq', self.GetFreq()
        print 'Address 0x%X' % self.usb_dev.ctrl_transfer(IN, 0x41, 0, 0, 1)[0]
        sm = self.usb_dev.ctrl_transfer(IN, 0x3B, 0, 0, 2)
        sm = UBYTE2.unpack(sm)[0]
        print 'Smooth tune', sm
      except:
        print "No permission to access the SoftRock USB interface"
        self.usb_dev = None

  def GetStartupFreq(self):	# return the startup frequency / 4
    if not self.usb_dev: return 0
    ret = self.usb_dev.ctrl_transfer(IN, 0x3C, 0, 0, 4)
    s = ret.tostring()
    freq = UBYTE4.unpack(s)[0]
    freq = int(freq * 1.0e6 / 2097152.0 / 4.0 + 0.5)
    return freq
    
  def GetFreq(self):	# return the running frequency / 4
    if not self.usb_dev: return 0
    ret = self.usb_dev.ctrl_transfer(IN, 0x3A, 0, 0, 4)
    s = ret.tostring()
    freq = UBYTE4.unpack(s)[0]
    freq = int(freq * 1.0e6 / 2097152.0 / 4.0 + 0.5)
    return freq    
    
  def SetFreq(self, freq):
    freq = int(freq/1.0e6 * 2097152.0 * 4.0 + 0.5)
    s = UBYTE4.pack(freq)
    try:
      self.usb_dev.ctrl_transfer(OUT, 0x32, si570_i2c_address + 0x700, 0, s)
    except usb.core.USBError:
      traceback.print_exc()
    
  def PTT(self, ptt):
    if self.usb_dev:
      try:
        self.usb_dev.ctrl_transfer(IN, 0x50, ptt, 0, 3)
      except usb.core.USBError:
        traceback.print_exc()
    

  def Quit(self):
    try:
      jacklib.deactivate(self.jackclient)
    except:
      pass
    try:
      jacklib.client_close(self.jackclient)
    except:
      pass
    

  def Sync(self):
    
    ## Find start by amplitude
    mv = 0.7 * self.iIa[self.rtframes:].max()
    sia = numpy.nonzero( self.iIa[self.rtframes:] > mv )[0]
    si = sia[0] + self.rtframes
    
    ## Phase change is after 100 frames, add a bit of a buffer
    synca = self.iIa[si:si+120] - 1j * self.iQa[si:si+120]    
    anglea = numpy.angle(synca,deg=True) + 180
    deltaaa = (anglea[1:] - anglea[:-1]) % 360
    
    ## Buffer FFT start window to 20 frames past phase change
    syncindex = (numpy.nonzero( (deltaaa > 90) & (deltaaa < 270) )[0][0]) + si + 20 
    
    self.synci = syncindex
    
    
  def DoFFT(self):
    
    ## Remove DC bias
    I = self.iIa[self.synci:self.synci+self.fftn]
    #I = I - numpy.mean(I)
    Q = self.iQa[self.synci:self.synci+self.fftn]
    #Q = Q - numpy.mean(Q)
    
    self.fftia[:] = I - 1j * Q
    if self.fftwindow != None: self.fftia[:] = self.fftwindow * self.fftia
    
    self.fft()

  def Test(self,iterations=10,sleep=None):
    
    print "Beginning Test",iterations,sleep
    
    for i in range(0,iterations):
      print "Test iteration",i
      self.M()
      self.Mprint()
      if sleep:
        time.sleep(sleep)      

  
  def JackProcess(self,nframes,arg):
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
    
  def Mprint(self,isdut=False):

    if self.printlevel > 0:
      print "Sync:%d" % self.synci,
      print "Freq:%d" % int(round(self.GetFreq()+self.freq)),
      cn = self.fftoa[self.fftbin]
      print "Real:%3.1f" % cn.real,
      print "Imag:%3.1f" % cn.imag,
      print "Mag:%3.1f" % numpy.abs(cn),
      print "Phase:%3.1f" % numpy.angle(cn,deg=True)
      
  def M(self,freq=None):
    
    if freq: self.SetFreq(freq-self.freq)
    
    self.PTT(1) 
    
    self.startframe = 0
    self.docapture.clear()
    self.docapture.wait()

    self.PTT(0)
    self.Sync()
    self.DoFFT()
    
  def M2Array(self,freq,array):
    i = 0
    for f in freq:
      self.M(int(f*1000000))
      array[i] = self.fftoa[self.fftbin]
      self.Mprint()
      i += 1    

  def MO(self,m):
    print "Beginning Open Measurements"
    self.M2Array(m.freq,m.open)
    
  def MS(self,m):
    print "Beginning Short Measurements"
    self.M2Array(m.freq,m.short)
    
  def ML(self,m):
    print "Beginning Load Measurements"
    self.M2Array(m.freq,m.load)

  def MD(self,m):
    print "Beginning DUT Measurements"
    self.M2Array(m.freq,m.dut)
    
  def SWR(self,m,iterations=100):
    if m.freq.size != 1:
      raise IndexError("SWR requires exactly 1 measurement")
    print "Beginning SWR Measurements"
    
    for j in range(0,iterations):
      self.M(int(m.freq[0]*1000000))
      m.dut[0] = self.fftoa[self.fftbin]
      m.PrintSWR()
 
  def PlotTD(self):
    fig = self.CreateFigure("Time Domain")
    sp = fig.add_subplot(111)
    xaxis = range(0,len(self.iIa))
    sp.plot(xaxis,self.iIa,'.-',color='b',label='iI')
    sp.plot(xaxis,self.iQa,'.-',color='r',label='iQ')
    sp.plot(xaxis,self.oIa,'.-',color='c',label='oI')
    sp.plot(xaxis,self.oQa,'.-',color='m',label='oQ')    
    sp.set_ylabel("Amplitude")
    sp.set_xlabel("Sample")
    #sp.legend(bbox_to_anchor=(1,-0.1))  
    sp.legend(loc=2,bbox_to_anchor=(0,-0.1),ncol=4)   
    plt.show()        
    
 
  
  def PlotFFTInput(self):
    fig = self.CreateFigure("FFT Input")
    sp = fig.add_subplot(111)
    xaxis = range(0,self.fftn)
    sp.plot(xaxis,[x.real for x in self.fftia],'.-',color='b',label='I')
    sp.plot(xaxis,[x.imag for x in self.fftia],'.-',color='r',label='Q')    
    sp.set_ylabel("Amplitude")
    sp.set_xlabel("Sample")
    #sp.legend(bbox_to_anchor=(1,-0.1))  
    sp.legend(loc=2,bbox_to_anchor=(0,-0.1),ncol=4)   
    plt.show()    
    

  def PlotFD(self,dbfs=True):
    freqspectrum = numpy.abs(self.fftoa)
    freqspectrum = numpy.concatenate( [freqspectrum[self.fftn/2:self.fftn],freqspectrum[0:self.fftn/2]] )
    if dbfs:
      zerodb = 20*numpy.log10(self.fftn/2)
      freqspectrum = (20*numpy.log10(abs(freqspectrum))) - zerodb
      
    fig = self.CreateFigure("Frequency Domain")
    sp = fig.add_subplot(111)
    xaxis = numpy.linspace(-self.Sr/2,self.Sr/2,self.fftn)
    sp.plot(xaxis,freqspectrum,'.-',color='k',label='Spectrum')
    sp.set_ylabel("Amplitude")
    sp.set_xlabel("Frequency")
    sp.legend(loc=2,bbox_to_anchor=(0,-0.1),ncol=4)   
    plt.show()          
      
    
  def CreateFigure(self,title):
    fig = plt.figure()
    fig.subplots_adjust(bottom=0.2)
    fig.suptitle(title, fontsize=20)
    return fig    
    
    
    
if __name__ == '__main__':
  a = VNA(fftn=1024,freq=3000.0)
  a.OpenSoftRock()
  a.Info()

