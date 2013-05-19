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



import scipy, jack, pylab, struct, usb.core, usb.util, traceback, pyfftw, numpy, time
import cPickle as pickle
import threading
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
  def __init__(self):
    
    ## State
    self.sync = True
    self.threaded = True
    self.printlevel = 1
    ## Create output array at 2343.75Hz as this will be centered in a bin for 48000Hz sampling rate
    ## Need buffer of 25ms RT delay via jack_delay (1200 samples at 48khz) + 21ms (1024 samples for FFT)
    ## Will create 10 buffers worth (256 buffer size) or 2560 samples
    self.freq = 2343.75
    self.fftbin = 50
    self.fftn = 1024
    self.rtt = 0.055
    self.Iamp=1.0
    self.Qamp=1.0
    self.Qphase=0.0


    jack.attach("vna")
    jack.register_port("iI", jack.IsInput)
    jack.register_port("iQ", jack.IsInput)
    jack.register_port("oI", jack.IsOutput)
    jack.register_port("oQ", jack.IsOutput)
    jack.activate()
    jack.connect("vna:oQ", "system:playback_1")
    jack.connect("vna:oI", "system:playback_2")
    jack.connect("system:capture_1", "vna:iQ")
    jack.connect("system:capture_2", "vna:iI")
    
    if self.sync:
      jack.register_port("iS", jack.IsInput)
      jack.register_port("oS", jack.IsOutput)      
      jack.connect("vna:oS", "system:playback_3")
      jack.connect("system:capture_3", "vna:iS")          

    self.frames = jack.get_buffer_size()
    self.Sr = float(jack.get_sample_rate())
    self.dt = 1.0/self.Sr

    self.rttsamples = int(self.rtt / self.dt)
    #self.arraylen = ((self.rttsamples + (2*self.fftn) + self.frames) / self.frames) * self.frames
    self.arraylen = (self.rttsamples / self.frames) * self.frames
    if (self.rttsamples % self.frames) != 0:
      self.arraylen = self.arraylen + self.frames

    self.capturei = self.arraylen - (self.fftn + 100)
     
    self.oa = self.ComplexSinusoid(self.freq,self.arraylen)
    if self.sync:
      self.ia = scipy.zeros( (3,self.arraylen) ).astype(scipy.float32)      
    else:
      self.ia = scipy.zeros( (2,self.arraylen) ).astype(scipy.float32)
    
    self.fftia = pyfftw.n_byte_align_empty(self.fftn, 16, 'complex128')
    self.fftoa = pyfftw.n_byte_align_empty(self.fftn, 16, 'complex128')
  
    ## Create FFT Plan
    self.fft = pyfftw.FFTW(self.fftia,self.fftoa)
    
    if self.threaded:
      self.docapture = threading.Event()
      self.docapture.set()
      self.jack_thread = threading.Thread(target=self.JackThread)
      self.jack_thread.setDaemon(True)
      self.jack_thread.start()   
      
  def Info(self):
    print "Freq:",self.freq,"FFTbin:",self.fftbin,"FFTn:",self.fftn,"RTT:",self.rtt,"RT Samples:",self.rttsamples
    print "ArrayLen:",self.arraylen,"CaptureIndex",self.capturei    
    
    
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
        print ('Startup freq', self.GetStartupFreq())
        print ('Run freq', self.GetFreq())
        print ('Address 0x%X' % self.usb_dev.ctrl_transfer(IN, 0x41, 0, 0, 1)[0])
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
    
  def __del__(self):
    jack.deactivate()
    jack.detach()
    
  def Quit(self):
    jack.deactivate()
    jack.detach()  
 
  def ComplexSinusoid(self,freq,samples,sst=0):
    sampling_times = self.dt * scipy.r_[sst:sst+samples]
    I = self.Iamp * scipy.cos( (2*scipy.pi*freq*sampling_times))
    ## Change sign of Qamp if needed
    Q = self.Qamp * scipy.sin( (2*scipy.pi*freq*sampling_times) + (2*scipy.pi*(self.Qphase/360.0)))
    #Q = Qamp * scipy.sin( (4*scipy.pi*freq*sampling_times) + (2*scipy.pi*(Qphase/360.0)))
    if self.sync:
      S = scipy.zeros( (len(sampling_times),), dtype=scipy.float32 )
      ## Apply first signal
      f = 1.0/(16*self.dt)
      syncsts = self.dt * scipy.r_[0:17]
      S[50:67] = scipy.sin(2*scipy.pi*f*syncsts)
      S[100+self.fftn:117+self.fftn] = scipy.sin(2*scipy.pi*f*syncsts)
      return scipy.array( [I.astype(scipy.float32),Q.astype(scipy.float32),S] )
    else:
      return scipy.array( [I.astype(scipy.float32),Q.astype(scipy.float32)] )
  
  def DoFFT(self):
    self.fftia[:] = self.ia[0,self.capturei:self.capturei+self.fftn] - 1j * self.ia[1,self.capturei:self.capturei+self.fftn]
    self.fft()

  def Test(self,iterations=10,sleep=None):
    
    print "Beginning Test",iterations,sleep
    
    for i in range(0,iterations):
      print "Test iteration",i
      self.M()
      self.Mprint()
      if sleep:
        time.sleep(sleep)      

  def TestTone(self,j=2000):
    sst = 0
    ia = scipy.zeros( (2,self.frames) ).astype(scipy.float32)    
    self.PTT(1)
    while j >= 0:
      oa = self.ComplexSinusoid(self.freq,self.frames,sst=sst)
      try:
        jack.process(oa,ia)
      except jack.InputSyncError:
        if sst != 0:
          print "InputSyncError"
      except jack.OutputSyncError:
        print "OutputSyncError"
      sst += self.frames
      j = j - 1
    self.PTT(0)
    
    
    
  def JackThread(self):
    
    if self.sync:
      chs = 3
    else:
      chs = 2
      
    dia = scipy.zeros( (chs,self.frames) ).astype(scipy.float32)
    doa = scipy.zeros( (chs,self.frames) ).astype(scipy.float32)
    
 
    while self.threaded:
      
      if not self.docapture.is_set():
        self.JackProcess()
        self.docapture.set()
      else:
        ## Service jack with default input and output
        try:
          jack.process(doa,dia)
          time.sleep(0.003)
        except jack.InputSyncError:
          if self.printlevel > 1:
            print "InputSyncError during default"
          pass
        except jack.OutputSyncError:
          if self.printlevel > 1:
            print "OutputSyncError during default"
          pass    


  def JackProcess(self):
    
    done = False
    while not done:
      done = True 
      i = 0
      while i <= self.oa.shape[1] - self.frames:
        try:
          jack.process(self.oa[:,i:i+self.frames], self.ia[:,i:i+self.frames])
          i += self.frames
        except jack.InputSyncError:
          ## Always expect an input sync error when first calling jack process
          if i != 0:
            if self.printlevel > 0: print "InputSyncError",i
            done = False
            break
          elif self.printlevel > 2:
            print "InputSyncError",i
            
        except jack.OutputSyncError:
          if self.printlevel > 0: print "OutputSyncError",i
          done = False
          break
          
      ## Test for proper sync
      if self.sync:
        ## FIXME: Problems if sample is just at edge
        sa = numpy.where( self.ia[2] > 0.2 )
        lsa = len(sa[0])
        if lsa % 2 != 0:
          if self.printlevel > 0: print "Uneven sync array",sa
          done = False
        else:
          fi = sa[0][0]
          si = sa[0][lsa/2]
          if ((si - fi) - 50) != self.fftn:
            if self.printlevel > 0: print "Incorrect number of samples between sync signals",fi,si
            done = False
          else:
            self.capturei = fi   
            
  def Mprint(self,isdut=False):

    if self.printlevel > 0:
      if self.sync:
        print "Sync:%d" % self.capturei,
      print "Freq:%d" % int(round(self.GetFreq()+self.freq)),
      cn = self.fftoa[self.fftbin]
      print "Real:%3.1f" % cn.real,
      print "Imag:%3.1f" % cn.imag,
      print "Mag:%3.1f" % numpy.abs(cn),
      print "Phase:%3.1f" % numpy.angle(cn,deg=True)
      
  def M(self,freq=None):
    
    if freq: self.SetFreq(freq-self.freq)
    
    self.PTT(1) 
    
    if self.threaded:
      self.docapture.clear()
      self.docapture.wait()
    else:
      self.JackProcess()

    self.PTT(0)
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
    pylab.plot(range(0,len(self.oa[0])),self.oa[0],'-o')
    pylab.plot(range(0,len(self.ia[0])),self.ia[0],'-o')
    pylab.show()
    
  def PlotSync(self):
    pylab.plot(range(0,len(self.oa[2])),self.oa[2],'-o')
    pylab.plot(range(0,len(self.ia[2])),self.ia[2],'-o')
    pylab.show()
    
  def PlotFFTWindow(self):
    pylab.plot(range(0,self.fftn),self.ia[0,self.capturei:self.capturei+self.fftn])
    pylab.plot(range(0,self.fftn),self.ia[1,self.capturei:self.capturei+self.fftn])
    pylab.show()
    

  def PlotFD(self,dbfs=True):
    freqspectrum = numpy.abs(self.fftoa)
    freqspectrum = numpy.concatenate( [freqspectrum[self.fftn/2:self.fftn],freqspectrum[0:self.fftn/2]] )
    if dbfs:
      zerodb = 20*numpy.log10(self.fftn/2)
      freqspectrum = (20*numpy.log10(abs(freqspectrum))) - zerodb
    pylab.plot(range(0,len(freqspectrum)),freqspectrum,'-o')
    pylab.show()
    
    
    
    
if __name__ == '__main__':
  a = VNA()
  a.OpenSoftRock()
  a.Info()

