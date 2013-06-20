
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
import numpy as np
import pickle, getopt, sys, os
import matplotlib as mpl
mpl.use('Qt4Agg',warn=False)
import matplotlib.pyplot as plt
import config


try:
  import smithplot
except:
  print("No pySmithChart available")
  


def LoadMeasurement(fn):
    '''Load a Saved Measurement Object'''
    pf = open(fn+".vam", 'rb')
    m = pickle.load(pf)
    pf.close()
    return m  


class Measurement:
  
  def __init__(self,fstart,fstop=None,fstep=0.1,eloadz=config.eloadz,feedlinez=config.feedlinez,view=False):
    """Create a Measurement Object"""

    ## Do not init anything if creating a view
    if view: return
    
    if fstop:
      ## Init frequency array
      self.freq = np.arange(fstart,fstop,fstep)
    else:
      self.freq = np.array( [fstart] )
    
    ## Measured open standard
    self.open = np.array( [0+0j]*self.freq.size )
    
    ## Measure short standard
    self.short = np.array( [0+0j]*self.freq.size )
    
    ## Measured load standard
    self.load = np.array( [0+0j]*self.freq.size )
    
    ## Measured DUT
    self.dut = np.array( [0+0j]*self.freq.size )

    ## Expected load standard Z
    self.eloadz = np.array( [eloadz]*self.freq.size )
    
    ## Impedance of line for calculating reflection coefficient
    self.feedlinez = np.array( [feedlinez]*self.freq.size )
    

  def __repr___(self):
    if self.freq.size > 1:
      return "Measurement(%s,%s,%s)" % (self.freq[0],self.freq[-1],self.freq[1]-self.freq[0])
    else:
      return "Measurement(%s)" % self.freq[0]
      
  ## Return a sliced view
  def __getitem__(self,sl):
    if self.freq == None:
      raise IndexError("No frequency array to index")
    elif self.freq.size == 1:
      raise IndexError("Can't slice measurement of size 1")
    
    fmin = self.freq[0]
    fmax = self.freq[-1]  
    fstep = self.freq[1] - fmin
    
    if isinstance(sl,slice):
      if sl.start:
        if fmin <= sl.start <= fmax:
          istart = int(np.floor((sl.start - fmin)/fstep))
        else:
          raise IndexError("Specified start frequency is out of bounds")
      else:
        istart = None
        
      if sl.stop:
        if (fmin <= sl.stop <= fmax) and (sl.stop>=sl.start):
          istop = int(np.ceil((sl.stop - fmin)/fstep))
        else:
          raise IndexError("Specified stop frequency is out of bounds")
      else:
        istop = None
        
      if sl.step:
        istep = int(np.floor(sl.step / fstep))
        if istep == 0: istep = None
      else:
        istep = None
      
      isl = slice(istart,istop,istep)
      
    else:
      
      if fmin <= sl <= fmax:
        i = int(np.round((sl - fmin)/fstep))
        isl = slice(i,i+1)
      else:
        raise IndexError("Specified frequency is out of bounds")

    ## Create view
    nms = Measurement(0,view=True)
    nms.freq = self.freq[isl]
    nms.open = self.open[isl]
    nms.short = self.short[isl]
    nms.load = self.load[isl]
    nms.eloadz = self.eloadz[isl]
    nms.feedlinez = self.feedlinez[isl]
    nms.dut = self.dut[isl]
    
    return nms
    
  def Copy(self):
    """Copy a Measurement Object"""
    nms = Measurement(0,view=True)
    nms.freq = np.copy(self.freq)
    nms.open = np.copy(self.open)
    nms.short = np.copy(self.short)
    nms.load = np.copy(self.load)
    nms.eloadz = np.copy(self.eloadz)
    nms.feedlinez = np.copy(self.feedlinez)
    nms.dut = np.copy(self.dut)
    return nms
    
 
  def Z(self):
    """Compute Complex Impedance"""
    return (self.eloadz * (self.open-self.load) * (self.dut-self.short)) / ((self.load-self.short) * (self.open-self.dut))
    
  def Rho(self,z=None):
    """Compute Reflection Coefficient"""
    if not z: z = self.Z()
    return (z - self.feedlinez)/(z + self.feedlinez)

  def SWR(self,rho=None):
    """Compute SWR"""
    if not rho: rho = self.Rho()
    rho = np.abs(rho)
    return (1+rho)/(1-rho)
    
  def RL(self,rho=None):
    """Compute Return Loss"""
    if not rho: rho = self.Rho()
    rho = np.abs(rho)
    return (-20 * np.log10(rho))
    
  def Strs(self,SWR=False,Z=False,RL=False,Rho=False,Open=False,Short=False,Load=False,ELoad=False,ZLine=False):
    ## Always include frequency
    fstrs = ['{:>9.6f} MHz'.format(x) for x in self.freq]
    li = [fstrs]
    
    if SWR: li.append(['SWR:{:>2.2f}'.format(x) for x in self.SWR()])
    if Z: li.append(['Z:({:>+7.2f}{:>+7.2f}j)'.format(x.real,x.imag) for x in self.Z()])
    if RL: li.append(['RL:{:>2.2f}'.format(x) for x in self.RL()])    
    if Rho: li.append(['Rho:({:>+7.2f}{:>+7.2f}j)'.format(x.real,x.imag) for x in self.Rho()])    
    if Open: li.append(['Open:({:>+7.2f}{:>+7.2f}j)'.format(x.real,x.imag) for x in self.open])
    if Short: li.append(['Short:({:>+7.2f}{:>+7.2f}j)'.format(x.real,x.imag) for x in self.short])        
    if Load: li.append(['Load:({:>+7.2f}{:>+7.2f}j)'.format(x.real,x.imag) for x in self.load])
    if ELoad: li.append(['ExpectedLoad:({:>+7.2f}{:>+7.2f}j)'.format(x.real,x.imag) for x in self.eloadz])
    if ZLine: li.append(['Zline:({:>+7.2f}{:>+7.2f}j)'.format(x.real,x.imag) for x in self.feedlinez])
    
    return zip(*li)
    
    
  def Print(self):
    strs = self.Strs(True,True,True,True,True,True,True,True,True)
    for s in strs:
      for i in s: print(i,)
      print()
      
  def PrintSWR(self):
    """Print SWR"""
    strs = self.Strs(SWR=True)
    for s in strs:
      for i in s: print(i,)
      print()
      
      
  def PlotRho(self):
    """Plot Rho"""
    fig = self.CreateFigure("Rho vs Frequency")
    sp = fig.add_subplot(111)
    self.SubPlotComplex(sp,self.Rho())
    self.SubPlotMHz(sp)
    plt.show()    
    
  def PlotOpen(self):
    """Plot Open Standard"""
    fig = self.CreateFigure("Open vs Frequency")
    sp = fig.add_subplot(111)
    self.SubPlotComplex(sp,self.open)
    self.SubPlotMHz(sp)
    plt.show()        
    
  def PlotLoad(self):
    """Plot Load Standard"""
    fig = self.CreateFigure("Load vs Frequency")
    sp = fig.add_subplot(111)
    self.SubPlotComplex(sp,self.load)
    self.SubPlotMHz(sp)
    plt.show()        

  def PlotDUT(self):
    """Plot DUT"""
    fig = self.CreateFigure("DUT vs Frequency")
    sp = fig.add_subplot(111)
    self.SubPlotComplex(sp,self.dut)
    self.SubPlotMHz(sp)
    plt.show()        


  def PlotZ(self,absxc=True):
    """Plot DUT Z vs Frequency"""
    fig = self.CreateFigure("DUT Z vs Frequency")
    sp = fig.add_subplot(111)
    self.SubPlotImpedance(sp,self.Z(),absxc)
    self.SubPlotMHz(sp)
    plt.show()        
    
  def PlotELoadZ(self,absxc=True):
    """Plot Expected Load Z vs Frequency"""
    fig = self.CreateFigure("Expected Load Z vs Frequency")
    sp = fig.add_subplot(111)
    self.SubPlotImpedance(sp,self.eloadz,absxc)
    self.SubPlotMHz(sp)
    plt.show()       
    
  def PlotFeedLineZ(self,absxc=True):
    """Plot Feedline Z vs Frequency"""
    fig = self.CreateFigure("Line Z vs Frequency")
    sp = fig.add_subplot(111)
    self.SubPlotImpedance(sp,self.feedlinez,absxc)
    self.SubPlotMHz(sp)
    plt.show()       
    
  def PlotSWR(self):
    """Plot SWR"""
    fig = self.CreateFigure("SWR vs Frequency")
    sp = fig.add_subplot(111)
    self.SubPlotSWR(sp)
    self.SubPlotMHz(sp)
    plt.show()        
    
  def PlotRL(self):
    """Plot Return Loss vs Frequency"""
    fig = self.CreateFigure("Return Loss vs Frequency")
    sp = fig.add_subplot(111)
    self.SubPlotRL(sp)
    self.SubPlotMHz(sp)
    plt.show()

  def Plot(self,absxc=True):
    fig = self.CreateFigure("DUT Z and SWR vs Frequency")
    sp1 = fig.add_subplot(111)
    self.SubPlotImpedance(sp1,self.Z(),absxc)
    sp2 = sp1.twinx()
    self.SubPlotSWR(sp2)
    self.SubPlotMHz(sp1)
    plt.show()       

  def CreateFigure(self,title):
    fig = plt.figure()
    fig.subplots_adjust(bottom=0.2)
    fig.suptitle(title, fontsize=20)
    return fig
 
  def SubPlotMHz(self,sp):
    ## Appropriate xscale
    fmin = self.freq[0]
    fmax = self.freq[-1]
    frange = fmax - fmin

    sp.set_xlim( (fmin,fmax) )

    sp.xaxis.set_major_locator(mpl.ticker.LinearLocator())
    
    if frange > 1.0:
      sp.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
    else:
      sp.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.3f"))      
    
    sp.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())        

    sp.yaxis.grid(True)
    sp.xaxis.grid(True)

    sp.tick_params(axis='x', pad=10)    
    sp.set_xlabel("MHz") 
 
 
  def SubPlotImpedance(self,sp,ca,absxc=True):    
    x = ca.imag
    nxi = np.where( x < 0)
    if absxc:
      xcstr = '|Xc|'
      nx = np.abs(x[nxi])
    else:
      xcstr = 'Xc'
      nx = x[nxi]
    nxf = self.freq[nxi]
    pxi = np.where( x >= 0)
    px = x[pxi]
    pxf = self.freq[pxi]
    rl = ca.real
 
    sp.plot(self.freq,rl,'.-',color='b',label='R')
    sp.plot(nxf,nx,'.-',color='r',label=xcstr)
    sp.plot(pxf,px,'.-',color='g',label='Xl')
    sp.plot(self.freq,np.abs(ca),'.-',color='y',label='Z')     
    sp.set_ylabel("Ohms")
    sp.legend(loc=2,bbox_to_anchor=(0,-0.1),ncol=4)
    

  def SubPlotComplex(self,sp,ca,yaxislabel=""):
    sp.plot(self.freq,ca.real,'.-',color='b',label='Real')
    sp.plot(self.freq,ca.imag,'.-',color='r',label='Imag')
    sp.set_ylabel(yaxislabel)
    sp.legend(loc=2,bbox_to_anchor=(0,-0.1),ncol=4)
    
  def SubPlotSWR(self,sp):
    sp.plot(self.freq,self.SWR(),'.-',color='c',label='SWR')
    sp.set_ylabel("SWR")
    sp.legend(bbox_to_anchor=(1,-0.1))
 
  def SubPlotRL(self,sp):
    
    sp.plot(self.freq,self.RL(),'.-',color='m',label='RL')
    sp.set_ylabel("dB")
    sp.legend(bbox_to_anchor=(1,-0.1))    
 
  def SaveLinSmith(self,fn):
    """Export LinSmith File"""
    linsmithf = open(fn+".load", 'w')
    linsmithf.write('<?xml version="1.0"?>\n')
    linsmithf.write("<loads>\n")
    
    for f,z in zip(self.freq,self.Z()):
      linsmithf.write(' <load f="%f" r="%f" x="%f" />\n' % (f,z.real,z.imag))
  
    linsmithf.write("</loads>\n")
    linsmithf.close()


  def SaveMeasurement(self,fn):
    """Save Measurement Object"""
    pkl_file = open(fn+".vam", 'wb')    
    pickle.dump( self, pkl_file )
    pkl_file.close()



  def PlotSmithChart(self):
    """Plot a Smith Chart of DUT Z"""
    # plot data
    plt.figure(figsize=(8, 8))

    ax = plt.subplot(1, 1, 1, projection='smith', axes_scale=np.abs(self.feedlinez[0]))

    plt.plot(self.Z(), path_interpolation=0, label="DUT")

    plt.legend(loc="lower right")
    plt.title("DUT Z Smith Chart")

    plt.show()
    
    
if __name__ == '__main__':
    
  def PrintUsage():
    print('python -i Measurement.py -f <filename>')
    
  try:
    opts, args = getopt.getopt(sys.argv[1:],"hf:",[])
  except getopt.GetoptError:
    PrintUsage()
    os._exit(2)

  for opt, arg in opts:
    if opt == '-h':
      PrintUsage()
      os._exit(0)
    elif opt == '-f':
      m = LoadMeasurement(arg)
      
    



