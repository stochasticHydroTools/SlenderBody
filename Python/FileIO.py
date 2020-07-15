"""
Utilities for file input and output
"""

def prepareOutFile(name):
    outFile = name;
    of = open(outFile,'w');
    of.close();
    of = open(outFile,'a');
    return of;

def writeArray(name,stress,wora='w'):
    """
    Write the locations to a file object of. 
    """
    outFile = name;
    of = open(outFile,wora);
    N = len(stress); # to save writing
    for i in range(N):
        of.write('%14.15f \n' % stress[i]);