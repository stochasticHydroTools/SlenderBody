import numpy as np
import SpecQuadUtils as sq
import EwaldNumba as ewNum
import EwaldUtils as ewc


""" 
These are the functions for the target and fiber velocity corrections, 
meaning it handles all of the special quadrature stuff. This class
only does free space corrections, it therefore DOES NOT need geometric
information.
"""


