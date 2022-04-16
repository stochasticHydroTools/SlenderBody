all: UAMMD_PSE BatchedNBodyRPY cppfiles Neighbors finufft

UAMMD_PSE:    
	cd Dependencies/UAMMD_PSE_Python && $(MAKE)

BatchedNBodyRPY:
	cd Dependencies/BatchedNBodyRPY && $(MAKE)
	
cppfiles: 
	cd cppmodules && $(MAKE)
	
Neighbors:
	cd Dependencies/NeighborSearch && $(MAKE)
	
finufft:
	cd Dependencies/finufft && $(MAKE) python3
	



