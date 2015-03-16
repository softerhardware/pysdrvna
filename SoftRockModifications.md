# SoftRock RXTX Ensemble Modifications #

Paul (N2PK) wrote a [tutorial](http://www.dxatlas.com/rocky/Files/Rocky_VNA_v1c.pdf) on modifying an early crystal-based [SoftRock v6.1](http://www.qrpradio.org/pub/softrocks/manuals/softrock%20v6.1%20rxtx) to work as a VNA. The modifications described here are an update of his work for the latest Si570-based [RXTX Ensemble](http://fivedash.com/) and should also work with [Rocky's VNA](http://www.dxatlas.com/rocky/Advanced.asp). Essentially, you must enable the receiver during transmit and optionally disable the final PA when in VNA mode. To add VNA mode,

  * Disconnect R 51 from QSD\_EN and pull QSD\_EN high with a similar value resistor. I used a 3k resistor. This will enable the receiver during transmit.
  * Optionally ground the center tap of T3, also identified as where R 47 and R 46 connect together. This will disable the PA during transmit.

I have my PA disabled but you may be able to reduce the power enough for clean measurements by decreasing the audio drive.

The RXTX Ensemble [schematics](http://www.wb5rvz.org) will help with these modifications. I added a RadioShack DPDT switch to my [KM5H enclosure](http://km5h.softrockradio.org/) for "Operate" and "VNA" modes.


<img src='http://pysdrvna.googlecode.com/files/vnamod.jpg' alt='Logo' />