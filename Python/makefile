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

clean:
	$(MAKE) -C Dependencies/BatchedNBodyRPY clean
	$(MAKE) -C Dependencies/UAMMD_PSE_Python clean
	$(MAKE) -C cppmodules clean
	$(MAKE) -C Dependencies/NeighborSearch clean
	$(MAKE) -C Dependencies/finufft clean
