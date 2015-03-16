# Accuracy #

To understand how accurate the pysdrvna is, I compared it to a DG8SAQ SDR-Kits V2 VNA. The conclusion is that with reasonable care, you can make reflection measurements that are within 2% of the DG8SAQ VNA. This is very accurate for amateur radio purposes. The prime causes of decreased accuracy are distortion in the receiver from too much drive, and heating of components from enabling PTT for extended periods.

Except where noted, the following experiments used:

  * An M-Audio Delta 66 sound card sampling at 48 kHz
  * The same set of near lab grade OSL standards, 50 Ohm Load standard
  * A 121 Ohm resistor in series with a ~380pF ceramic capacitor test standard. Both were surface mount devices soldered to a BNC connector.
  * Test at 10.130 MHz
  * Plots are of the corrected reflection coefficient magnitude and phase
  * 500 repeated measurements


## 1. Jack Latency and FFT Size ##

The first test used a fairly large FFT, N=2048, and long jack latency, 42.7ms. The jack settings were 1024 frames/period and 2 periods/buffer.

The red plot shows the change in the reflection angle over the 500 measurements made by pysdrvna. The red bar shows the angle range the DG8SAQ VNA measured in 10 trials, with the highest and lowest thrown out. Similarly, the blue line shows the change in the reflection magnitude over the 500 measurements made by pysdrvna. The DQ8SAQ VNA only provided resolution to the hundredths place for magnitude measurements and hence all reported magnitude measurements were 0.47, the center of the left vertical axis. The extents of the left and right vertical axis go to +/- 1% error. Therefore, from visual inspection, all pysdrvna measurements for this test are within 1% of the DG8SAQ VNA.

<img src='http://pysdrvna.googlecode.com/files/atest1.png' />

## 2. Reduced Jack Latency ##

As seen in the first experiment, the reflection coefficient diverges over time from the DG8SAQ VNA due to heating of components. (This heating occurs whenever PTT is enabled, even if there is no audio drive for transmit.) PTT is enabled for the duration of a single measurement. This time can be reduced by reducing jack's latency. My system can go down to 2.67ms latency with 64 frames/period.

Note that pysdrvna can tolerate an occasional xrun, as it will detect it and retry. Even though pysdrvna is written in Python, it does no processing during the audio collection phase and hence does not cause xruns. On my system, most of the xruns are caused by my task panel auto hiding!

The following graph shows similar character to the last graph. All measurements are within 1% of the DG8SAQ VNA. No significant differences are seen. What is significant, is that the 500 measurements complete in much less time!

<img src='http://pysdrvna.googlecode.com/files/atest2.png' />

## 3. Smaller FFT ##

Again, to reduce the time PTT is enabled, we can make the FFT smaller. This experiment matches the last experiment, except the size of the FFT has been reduced to N=512.

Note that there is more variation in the measured reflection angle. Also note that the measurements do not diverge as much. These measurements are well within 1% of the DG8SAQ VNA. This is still a negligible accuracy improvement under typical use, but it does help us understand what does or does not effect accuracy.

<img src='http://pysdrvna.googlecode.com/files/atest3.png' />


## 4. Warm Up and Cool Down Times ##

In this experiment, a warm up time of 0.3 seconds is specified. This can be set in the config.py file. This is to address the rapid heating seen for the first few measurements. Also, a cool down time of 0.07 seconds is added between measurements. This is to mitigate the effects of heating. Finally, the FFT size is increased to 1024 to reduce variation in the measured angle as well as provide narrower frequency bins when conected to a real antenna with real interference.

Note that these settings are the best so far.

<img src='http://pysdrvna.googlecode.com/files/atest4.png' />


## 5. Amplitude Variation with PA Disabled ##

By default, the PA is disabled on my RXTX Ensemble when using it as a VNA. I adjust the amplitude using the Envy24 mixer so that the pysdrvna test tone is at around -20 dbFS as seen in the frequency domain plot produced by vna.PlotFD(). In this plot, the image is around -60 dbFS. Sometimes I see an additional 1 or 2 spikes at around -80 dbFS. What effect does changing this amplitude have?

### a. Low Amplitude ###

The plot shown below lowers the amplitude so that the main spike is around -40 dbFS. As can be seen, the measurements are still good. You will want to run at a higher amplitude to minimize effects from strong signals when connected to a real antenna.

<img src='http://pysdrvna.googlecode.com/files/atest5a.png' />


### b. High Amplitude ###

At high amplitudes (main spike at -5 dbFS and -10 dbFS in vna.PlotFD()) the SoftRock receiver overloads and produces a distorted signal. This causes the readings from the pysdrvna to be bogus. I can not even say if the readings are off by 10% or 20% as they just do not make sense. You can check if you amplitude is to high by using vna.PlotFD() and vna.PlotTD(). In the frequency domain, you should see only the main spike and perhaps and image. In the time domain, you should see clean sine waves received.


### c. Highest Working Amplitude ###

The following plot shows the highest working amplitude, main spike at -12 dbFS. Although the measurements are within 1% of the DG8SAQ VNA, you can see that they are diverging faster than before. Also, you will find that the main spike amplitude varies with frequency and connected load. I recommend calibrating your main spike to about -20 dbFS to accommodate these fluctuations.

<img src='http://pysdrvna.googlecode.com/files/atest5c.png' />


## 6. Power Amplifier Enabled ##

I enabled my PA when in VNA mode to see what effect it would have. It was much trickier to find a amplitude setting that produced reasonable results. I had to lower the amplitude via the Envy24 mixer as well set the config.py amp parameter to 0.1. In the plot below, the vertical axis extents now represent a **+/- 2%** difference from the DG8SAQ VNA. The measurements exhibit more variation but are still usable.

<img src='http://pysdrvna.googlecode.com/files/atest6.png' />

## 7. Test Tone Frequency ##

By default, pysdrvna uses a single test tone at 2350 Hz. What effect does varying the frequency of the test tone have? The following plots show little variation up to 10000 Hz. However, past 10000 Hz there are problems with pysdrvna detecting the phase reversal sync and measurements become bogus.

### a. Test tone at 1000 Hz ###

<img src='http://pysdrvna.googlecode.com/files/atest7b.png' />

See Section 4 for test tone at 2350 Hz

### b. Test tone at 5000 Hz ###

<img src='http://pysdrvna.googlecode.com/files/atest7c.png' />

### c. Test tone at 10000 Hz ###

<img src='http://pysdrvna.googlecode.com/files/atest7d.png' />


## 8. Measurement Frequency ##

So far, all the experiments have been done at an operating frequency of 10.130 MHz. How does pysdrvna compare to a DG8SAQ VNA at other frequencies? Below are plots for measuring the same standard at 7.13 MHz and 14.13 MHz. Note that in these plots the vertical axis extents now represent a **+/- 2%** difference from the DG8SAQ VNA. As can be seen, these measurements are not as close as the 10.13 MHz measurements. In particular, the 14.13 MHz angle measurements disagree by more than 2% after about 200 measurements. The heating effect appears more pronounced at 14.13 MHz. Still, in typical use where fewer than 200 measurements are made between calibrations, the disagreement is less than 2%.


### a. Measurements at 7.13 MHz ###

<img src='http://pysdrvna.googlecode.com/files/atest8_40.png' />



### b. Measurements at 14.13 MHz ###

<img src='http://pysdrvna.googlecode.com/files/atest8_20.png' />


## 9. Sound Card ##

What effect does the sound card quality have on measurements? A good USB sound card (Creative Sound Blaster X-Fi USB) and the built in sound of my inexpensive and old Acer AspireOne netbook are tested in this experiment. Jack was set to 8ms latency, 16 bit, with the USB sound card, and 5.33ms, 24 bit, with the netbook sound card. Although you see cleaner results with better sound cards, all the sounds cards I tried produced usable measurements.

### a. Sound Blaster X-Fi USB ###

<img src='http://pysdrvna.googlecode.com/files/atest9_usb.png' />


### b. Netbook Sound ###

<img src='http://pysdrvna.googlecode.com/files/atest9_netbook.png' />


## 10. Sample OSL Standards 500 Times ##

So far, all experiments have samples the Open, Short and Load standards just once, and then used those measurements for all 500 device measurements. It is easy to also sample the OSL standards 500 times so that these measurements are effected by heat in the same way as the device measurements.

<img src='http://pysdrvna.googlecode.com/files/atest10.png' />



## 11. Different Load Value ##

All experiments so far have used a load standard of 50 Ohms. Other load values are possible and pysdrvna will account for them correctly in calculations. My homemade load standard is 60.3 Ohms. The plot below shows the same test standard measured after calibrating with my homemade set of standards.  Results are within 2% for this plot.

<img src='http://pysdrvna.googlecode.com/files/atest11.png' />


## 12. Reuse OSL Measurements After 24 Hours ##

Good measurements are possible even when reusing a measurement object that was calibrated 24 hours in the past. The plot below are measurements made after my RXTX Ensemble had been off for ~24 hours but reusing the single OSL measurements made the day before.

<img src='http://pysdrvna.googlecode.com/files/atest12.png' />


## 13. Tweak FFT Window Start ##

The current config.py file sets the beginning of the FFT window to occur 50 samples after the phase reversal sync detection. If you look at the time domain plot (vna.PlotTD()), you will see that this is a reasonable amount of settling time for a stable signal. I did try varying this, but found no significant effect. You just want to stay away from any variation that occurs at the beginning of the receive window.












