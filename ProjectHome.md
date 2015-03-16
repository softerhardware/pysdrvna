# Introduction #

PySDRVNA is a Python toolkit to use a Software Defined Radio (SDR) as a simple Vector Network Analyzer (VNA). Its primary use is amateur radio antenna analysis for the HF frequencies covered by the SDR. You use it to collect complex impedance measurements and create plots such as those seen below for my 30M magnetic loop antenna.

<img src='http://pysdrvna.googlecode.com/files/MagLoop30M.png' alt='Logo' />

<img src='http://pysdrvna.googlecode.com/files/smithchart.png' alt='Logo' />


## Summary ##

  * Inspired by [Rocky's VNA](http://www.dxatlas.com/rocky/Advanced.asp) but for Linux
  * Support for SoftRock RXTX Ensemble with minor modifications
  * Interactive and programmatic Python interface
  * Complex impedance, reflection coefficient, SWR and return loss measurements
  * Sweep across entire frequency range of SoftRock RXTX Ensemble
  * Matplotlib for plotting
  * JACK for SDR/Audio interface
  * Numpy for data structures and DSP
  * FFTW3 for DSP


## Getting Started ##
  * SoftRockModifications: Modify your SoftRock RXTX Ensemble to be a VNA
  * [OSLStandards](OSLStandards.md): Create open short and load standards to calibrate your VNA
  * [PySDRVNA Tutorial](https://pysdrvna.googlecode.com/files/PySDRVNA_Tutorial.pdf): How to use the software
  * [Accuracy](Accuracy.md): Understand how to get the most accurate measurements