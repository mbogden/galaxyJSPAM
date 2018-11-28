"""galaxy.py
Author : Jackson Cole
Affil  : Middle Tennessee State University
Purpose: This class is to be used with the MergerRun class. It is
         useful for encapsulating information pertaining to the primary
         and secondary disks in a particular merger morphology.
"""

class Galaxy:
    def __init__(self, name, particles, ra, dec, xc, yc):
        """Constructor for Galaxy class
        """
        self.name         = name
        self.particles    = particles
        self.ra           = ra
        self.dec          = dec
        self.xc           = xc
        self.yc           = yc
