##############################################
## Edit VNA defaults here

## Length of FFT to use
fftn = 1024

## Test tone target frequency. Will be adjusted to center of nearest FFT bin.
#freq = 10000.0
freq = 2350.0

## Test tone amplitude
amp = 1.0
#amp = 0.12

#### Timing parameters
## Round trip time in samples from start of test tone to when test tone is received
## Typically set this to None if unknown
## After a test, you can and should execute ResizeArrays() to set this tightly
rtframes = None  

## Time in samples from start of test tone to when phase reversal sync happens
rtframes2sync = 50

## Time in samples from phase reversal sync to start of FFT window
sync2fft = 50

## Time in smaples from end of FFT window to end of test tone. 
fft2end = 10


#### Jack connection information
## I input channel
inI = "system:capture_2"

## Q input channel
inQ = "system:capture_1"

## I output channel
outI = "system:playback_2"

## Q output channel
outQ = "system:playback_1"


#### Measurement defaults

## The true value of your load standard.
## This does not need to be 50O Ohms, but should be close.
#eloadz = 60.3+0j
eloadz = 50.0+0j

## The Z of your feedline. Typically 50 Ohms.
feedlinez = 50+0j


#### Specify how Reflection Coefficient Gamma and Z are computed
## Set True to Compute Z from Gamma, otherwise False to directly compute from measurement data using ABCD parameters
ZfromGamma = True

## Set True to Compute Gamma from Z, otherwise False to directly compute from measurement using S11 correction
GammafromZ = False


#### Heating/stability defaults
## The time in seconds to warm up before making a measurement or series of measurements
## Specify None if no warmuptime is desired
## Warmup time in range of 1 second works well
warmuptime = 0.3

## The time in seconds to cooldown between measurement in a series of measurements
## Specify None if no cooldown time
## Cooldowntime of 0.06-0.09 works well
cooldowntime = 0.07


## Print verbosity
## 0 is none, 1 and higher increases verbosity
printlevel = 1
