#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <chrono>
#include "types.h"
#include "VectorMethods.cpp"
#include<iostream>
/**
Documentation last updated: 03/12/2021
C++ class for a cross-linked network where we track each end separately
This class is bound to python using pybind11
See the end of this file for bindings.

There are 4 reactions
1) Binding of a floating link to one site (rate _kon)
2) Unbinding of a link that is bound to one site to become free (reverse of 1, rate _koff)
3) Binding of a singly-bound link to another site to make a doubly-bound CL (rate _konSecond)
4) Unbinding of a double bound link in one site to make a single-bound CL (rate _koffSecond)
**/

extern "C"{ // functions from fortran MinHeapModule.f90 for the heap that manages the events

    void initializeHeap(int size);
    void deleteHeap();
    void increaseHeapSize();
    void resetHeap();
    void insertInHeap(int elementIndex, double time);
    void deleteFromHeap(int elementIndex);
    void updateHeap(int elementIndex,double time);
    void testHeap();
    void topOfHeap(int &index, double &time);

}

namespace py=pybind11;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> npInt;
typedef py::array_t<double, py::array::c_style | py::array::forcecast> npDoub;

class EndedCrossLinkedNetwork {

    public:

    EndedCrossLinkedNetwork(int TotSites, int maxLinksPerSite, vec rates, vec CLBounds,double kT,
        double restlen, double KStiffness,double CLseed){
        _TotSites = TotSites;
        _kon = rates[0];
        _konSecond = rates[1];
        _koff = rates[2];
        _koffSecond = rates[3];
        _FreeLinkBound = intvec(TotSites,0);
        _TotalNumberBound = intvec(TotSites,0);
        _MaxNumberPerSite = maxLinksPerSite;
        _maxLinks = std::max(4*int(_konSecond/_koffSecond*_kon/_koff*_TotSites),100); // guess for max # of links
        //std::cout << "Array size " << _maxLinks << std::endl;
        _LinkHeads = intvec(_maxLinks,-1);
        _LinkTails = intvec(_maxLinks,-1);
        _RealDistances = vec(_maxLinks,-1);
        _LinkShiftsPrime = vec(3*_maxLinks,0); // these are in the primed coordinate space
        initializeHeap(_maxLinks+_TotSites+4);
        rng.seed(CLseed);
        unif = std::uniform_real_distribution<double>(0.0,1.0);
        _nDoubleBoundLinks = 0;
        _lowerCLBound = CLBounds[0];
        _upperCLBound = CLBounds[1];
        _kT = kT;
        _restlen = restlen;
        _KStiffness = KStiffness;
        _UnloadedRate = 0;
        _LockContract = false;
        _SpatialBinding = false;
     }
     
     void SetMotorParams(double UnloadedRate, double StallForce, int SitesPerFib){
        _UnloadedRate=UnloadedRate;
        _FStall = StallForce;
        _NSitesPerFib = SitesPerFib;
     }
     
     void ChangeLockContract(bool which){
        _LockContract = which;
    }
    
    void ChangeSpatialBinding(bool which,double SinFreq){
        _SpatialBinding = which;
        _SinFreq = SinFreq;
    }

     ~EndedCrossLinkedNetwork(){
        deleteHeap();
     }


     void updateNetwork(double tstep, npInt pyMaybeBindingPairs, npDoub pyuniPts, 
        npDoub pyPrimeShiftProposed, npDoub pyRealShiftProposed,npDoub pyRealShift){
        /*
        Update the network using Kinetic MC.
        Inputs: tstep = time step. pyMaybeBindingPairs = 2D numpy array, where the first
        column is the bound end and the second column is the unbound end, of possible
        link binding site pairs.
        pyuniPts = 2D numpy array of uniform points. g = strain in coordinate system
       
        */
        // Convert numpy to C++ vector and reset heap for the beginning of the step
	    intvec MaybeBindingPairs(pyMaybeBindingPairs.size());
        std::memcpy(MaybeBindingPairs.data(),pyMaybeBindingPairs.data(),pyMaybeBindingPairs.size()*sizeof(int));
        vec uniPts(pyuniPts.size());
        std::memcpy(uniPts.data(),pyuniPts.data(),pyuniPts.size()*sizeof(double));
        vec PrimeShiftProposed(pyPrimeShiftProposed.size());
        std::memcpy(PrimeShiftProposed.data(),pyPrimeShiftProposed.data(),pyPrimeShiftProposed.size()*sizeof(double));
        vec RealShiftProposed(pyRealShiftProposed.size());
        std::memcpy(RealShiftProposed.data(),pyRealShiftProposed.data(),pyRealShiftProposed.size()*sizeof(double));
        vec RealShift(pyRealShift.size());
        std::memcpy(RealShift.data(),pyRealShift.data(),pyRealShift.size()*sizeof(double));
        resetHeap();

        // Determine pairs that can actually bind
        int nPropUnique = MaybeBindingPairs.size()/2;
        vec PropDistances(nPropUnique,-1);
        vec xBar(nPropUnique,-1);
        intvec FirstLinkBySite(_TotSites,-1);
        intvec NextLinkByLink(2*nPropUnique,-1);
        for (int iPair=0; iPair < nPropUnique; iPair++){
            PropDistances[iPair] = LinkDistance(MaybeBindingPairs[2*iPair],MaybeBindingPairs[2*iPair+1],uniPts,iPair,RealShiftProposed,xBar[iPair]);
            double BaseRate = BindingRate(PropDistances[iPair]);
            if (_SpatialBinding){
                BaseRate*=abs(sin(_SinFreq*xBar[iPair]));
            }   
            for (int End=0; End < 2; End++){
                int BoundEnd = MaybeBindingPairs[2*iPair+End];
                TimeAwareHeapInsert(2*iPair+End+1, BaseRate*_FreeLinkBound[BoundEnd],0,tstep);// notice we start indexing from 1, this is how fortran is written
                // Create linked list of pairs by site so that the rates can be updated rapidly when there are new bound ends 
                if (FirstLinkBySite[BoundEnd]==-1){
                    FirstLinkBySite[BoundEnd]=2*iPair+End;  
                } else {
                    int CurLink = FirstLinkBySite[BoundEnd];
                    while (NextLinkByLink[CurLink] !=-1){
                        CurLink = NextLinkByLink[CurLink];
                    } 
                    NextLinkByLink[CurLink]=2*iPair+End;
                }
            }
        }
        // Single binding to a site
        double RateFreeBind = _kon*_TotSites;
        int indexFreeBinding = 2*nPropUnique+1;
        TimeAwareHeapInsert(indexFreeBinding, RateFreeBind,0,tstep);

        // Single unbinding from a site
        int indexFreeUnbinding = indexFreeBinding+1;
        for (int iSite = 0; iSite < _TotSites; iSite++){
            double UnbindRate = _koff*_FreeLinkBound[iSite];
            TimeAwareHeapInsert(indexFreeUnbinding+iSite, UnbindRate,0,tstep);
        }

        // One end of CL unbinding
        int indexSecondUnbind = indexFreeUnbinding+_TotSites;
        for (int iLink=0; iLink < _nDoubleBoundLinks; iLink++){
            double xBarThis=0;
            _RealDistances[iLink] = LinkDistance(_LinkHeads[iLink],_LinkTails[iLink],uniPts,iLink,RealShift,xBarThis);
            double UnbindRate=2*UnbindingRate(_RealDistances[iLink]);
            TimeAwareHeapInsert(indexSecondUnbind+iLink,UnbindRate,0,tstep);
        }

        double systime;
        int eventindex, BoundEnd, UnboundEnd;
        topOfHeap(eventindex,systime);
        //std::cout << "Top of heap is at index " << eventindex << " and time " << systime << std::endl;
        while (eventindex > 0) {
            if (eventindex==indexFreeBinding){ // end binding, choose random site and bind an end
                //std::cout << "Index " << eventindex << ", single binding at time " << systime << std::endl;
                BoundEnd = int(unif(rng)*_TotSites);
                // End can only bind if the site isn't full
                if (_TotalNumberBound[BoundEnd] < _MaxNumberPerSite){
                    _FreeLinkBound[BoundEnd]++;
                    _TotalNumberBound[BoundEnd]++;
                } // otherwise nothing happens
                TimeAwareHeapInsert(indexFreeBinding,RateFreeBind,systime,tstep);
            } else if (eventindex >= indexFreeUnbinding && eventindex < indexSecondUnbind){ // single end unbind
                BoundEnd = eventindex-indexFreeUnbinding;
                //std::cout << "Index " << eventindex << ", single unbinding at time " << systime << std::endl;
                _FreeLinkBound[BoundEnd]--;
                _TotalNumberBound[BoundEnd]--;  // that end loses a link
            } else if (eventindex >= indexSecondUnbind){ // CL unbinding
                int linkNum = eventindex - indexSecondUnbind;
                BoundEnd = _LinkHeads[linkNum];
                UnboundEnd = _LinkTails[linkNum];
                if (unif(rng) < 0.5){
                    BoundEnd = _LinkTails[linkNum];
                    UnboundEnd = _LinkHeads[linkNum];
                }
                deleteLink(linkNum); // this will replace the link by the last one in the list
                double UnbindRateNewLink = 2*UnbindingRate(_RealDistances[linkNum]);
                TimeAwareHeapInsert(eventindex,UnbindRateNewLink,systime,tstep); // compute new 
                TimeAwareHeapInsert(indexSecondUnbind+_nDoubleBoundLinks,0,systime,tstep); // remove last one
                _TotalNumberBound[UnboundEnd]--; // Only the unbound end loses a link
                // Add a free end at the other site. Always assume the remaining bound end is at the left
                //std::cout << "Index " << eventindex << "link num " << linkNum << " sites " << BoundEnd << " , " << UnboundEnd <<
                //    ", CL end unbind at time " << systime << std::endl;
                _FreeLinkBound[BoundEnd]++;
            } else { // CL binding
                // The index now determines which pair of sites the CL is binding to
                // In addition, we always assume the bound end is the one at the left
                int iPair = (eventindex-1)/2;
                double ShiftSign=1.0;
                BoundEnd = MaybeBindingPairs[2*iPair]; // eventindex-1=2*iPair+End
                UnboundEnd = MaybeBindingPairs[2*iPair+1];
                if (((eventindex-1) % 2)==1){
                    BoundEnd = MaybeBindingPairs[2*iPair+1]; // eventindex-1=2*iPair+End
                    UnboundEnd = MaybeBindingPairs[2*iPair];
                    ShiftSign=-1.0;
                }    
                //std::cout << "Index " << eventindex << ", CL link " << iPair << " both ends bind at time " << systime << std::endl;
                // Link can only bind if the unbound end is available
                if (_TotalNumberBound[UnboundEnd] < _MaxNumberPerSite){
                    _LinkHeads[_nDoubleBoundLinks] = BoundEnd;
                    _LinkTails[_nDoubleBoundLinks] = UnboundEnd;
                    _RealDistances[_nDoubleBoundLinks] = PropDistances[iPair];
                    // Add the unbinding event
                    double UnbindRateNewLink = 2*UnbindingRate(PropDistances[iPair]);
                    TimeAwareHeapInsert(indexSecondUnbind+_nDoubleBoundLinks,UnbindRateNewLink,systime,tstep);
                     // unbound end picks up a link, bound end loses a free link
                    _TotalNumberBound[UnboundEnd]++;
                    _FreeLinkBound[BoundEnd]--; // one less free link bound
                    for (int d=0; d < 3; d++){
                        _LinkShiftsPrime[3*_nDoubleBoundLinks+d] = ShiftSign*PrimeShiftProposed[3*iPair+d];
                    }
                    _nDoubleBoundLinks++;
                } // otherwise nothing happens
            }
            if (_nDoubleBoundLinks == _maxLinks){ // double size of link arrays if necessary
                _maxLinks*=2;
                _LinkHeads.resize(_maxLinks);
                _LinkTails.resize(_maxLinks);
                _LinkShiftsPrime.resize(3*_maxLinks);
                _RealDistances.resize(_maxLinks);
                //std::cout << "Expanding array size to " << _maxLinks << std::endl;
            }
            // Rates of CL binding change based on number of bound ends
            updateSecondBindingRate(BoundEnd, FirstLinkBySite, NextLinkByLink, PropDistances, xBar, systime, tstep);

            // Update unbinding event at BoundEnd (the end whose state has changed)
            //std::cout << "About to insert unbinding single end in heap with index " << BoundEnd+indexFreeUnbinding <<  std::endl;
            TimeAwareHeapInsert(BoundEnd+indexFreeUnbinding,_koff*_FreeLinkBound[BoundEnd],systime,tstep);

            topOfHeap(eventindex,systime);
            //std::cout << "[" << eventindex << " , " << systime << "]" << std::endl;
            // Debugging check
            /*intvec TotNum2(_TotSites, 0);
            for (int iSite = 0; iSite < _TotSites; iSite++){
                TotNum2[iSite] = _FreeLinkBound[iSite];
            }
            for (int iLink = 0; iLink < _nDoubleBoundLinks; iLink++){
                TotNum2[_LinkHeads[iLink]]++;
                TotNum2[_LinkTails[iLink]]++;
            }
            for (int iSite = 0; iSite < _TotSites; iSite++){
                if (TotNum2[iSite] != _TotalNumberBound[iSite]){
                    std::cout << "COUNTING ER!" << std::endl;
                    std::cout << "Tot number bound at site " << iSite << " = " << _TotalNumberBound[iSite] << std::endl;
                    std::cout << "Compared to " << iSite << " = " << TotNum2[iSite] << std::endl;
                    return;
                }
                if (_TotalNumberBound[iSite] > _MaxNumberPerSite){
                    std::cout << "EXCEED MAX!" << std::endl;
                }
            }*/
        }
    }
    
    npDoub MotorSpeeds(npDoub pyuniPts, npDoub pyuniTanVecs, npDoub pyRealShifts){
    
        vec uniPts(pyuniPts.size());
        std::memcpy(uniPts.data(),pyuniPts.data(),pyuniPts.size()*sizeof(double));
        vec uniTanVecs(pyuniTanVecs.size());
        std::memcpy(uniTanVecs.data(),pyuniTanVecs.data(),pyuniTanVecs.size()*sizeof(double));
        vec RealShifts(pyRealShifts.size());
        std::memcpy(RealShifts.data(),pyRealShifts.data(),pyRealShifts.size()*sizeof(double));
        // Walking of doubly-bound links (finite force)
        vec RateMove(2*_nDoubleBoundLinks); // might not be needed
        for (int iLink=0; iLink < _nDoubleBoundLinks; iLink++){
            // Compute the force
            vec ParallelForces = ComputeParForce(iLink,uniPts,uniTanVecs,RealShifts);
            double RateMove_i = std::max(_UnloadedRate*(1+ParallelForces[0]/_FStall),0.0); // moving velocity
            double RateMove_j = std::max(_UnloadedRate*(1+ParallelForces[1]/_FStall),0.0); // moving velocity
            RateMove[2*iLink]=RateMove_i;
            RateMove[2*iLink+1]=RateMove_j;
            //std::cout << "Rates of movement link " << iLink << " = " << RateMove[2*iLink] << " , " << RateMove[2*iLink+1] << std::endl;
            // Add to heap
        }
        return make1DPyArray(RateMove);
    }
        
    
    void WalkLinks(double tstep, npDoub pyuniPts, npDoub pyuniTanVecs, npDoub pyRealShifts){       
        // Not yet figured out: what happens when a link is sitting on the end. It 
        // can't move, but could it unbind? Need to figure that out (shouldn't be too 
        // big a deal to take it off)
        // Convert numpy to C++ vector and reset heap for the beginning of the step
	    vec uniPts(pyuniPts.size());
        std::memcpy(uniPts.data(),pyuniPts.data(),pyuniPts.size()*sizeof(double));
        vec uniTanVecs(pyuniTanVecs.size());
        std::memcpy(uniTanVecs.data(),pyuniTanVecs.data(),pyuniTanVecs.size()*sizeof(double));
        vec RealShifts(pyRealShifts.size());
        std::memcpy(RealShifts.data(),pyRealShifts.data(),pyRealShifts.size()*sizeof(double));
        resetHeap();

        // Walking of doubly-bound links (finite force)
        vec RateMove(2*_nDoubleBoundLinks); // might not be needed
        for (int iLink=0; iLink < _nDoubleBoundLinks; iLink++){
            // Compute the force
            vec ParallelForces = ComputeParForce(iLink,uniPts,uniTanVecs,RealShifts);
            double RateMove_i = std::max(_UnloadedRate*(1+ParallelForces[0]/_FStall),0.0); // moving velocity
            double RateMove_j = std::max(_UnloadedRate*(1+ParallelForces[1]/_FStall),0.0); // moving velocity
            RateMove[2*iLink]=RateMove_i;
            RateMove[2*iLink+1]=RateMove_j;
            //std::cout << "Rates of movement link " << iLink << " = " << RateMove[2*iLink] << " , " << RateMove[2*iLink+1] << std::endl;
            // Add to heap
            TimeAwareHeapInsert(2*iLink+1, RateMove_i,0,tstep);
            TimeAwareHeapInsert(2*iLink+2, RateMove_j,0,tstep);
        }
        
        // Walking of singly-bound link
        int UnloadedStart = 2*_nDoubleBoundLinks+1;
        for (int iSite = 0; iSite < _TotSites; iSite++){
            TimeAwareHeapInsert(UnloadedStart+iSite, _UnloadedRate*_FreeLinkBound[iSite],0,tstep);
        }

        double systime;
        int eventindex;
        topOfHeap(eventindex,systime);
        //std::cout << "Top of heap is at index " << eventindex << " and time " << systime << std::endl;
	    while (eventindex > 0) {
            //std::cout << "System time " << systime << std::endl;
            //std::cout << "Event index " << eventindex << std::endl;
            if (eventindex < UnloadedStart){ // Moving loaded links
                int iLink = (eventindex-1)/2; 
                bool IsTail = (eventindex % 2)==0;
                //std::cout << "Index (starts from 1!) " << eventindex << std::endl;
                int PtToMove = _LinkHeads[iLink];
                if (IsTail){
                    PtToMove = _LinkTails[iLink];
                }
                //std::cout << "PtToMove " << PtToMove << std::endl;
                // Check that it can move to the next site
                int LocalMovedIndex = (PtToMove+1) % _NSitesPerFib;
                bool IsEnd = LocalMovedIndex==0;
                bool IsFull = false;
                if (!IsEnd){
                    IsFull = _TotalNumberBound[PtToMove+1]==_MaxNumberPerSite;    
                }
                bool pastHalf = _LockContract && (LocalMovedIndex > _NSitesPerFib/2); // TEMP to lock in contractile configurations
                if (IsEnd || IsFull || pastHalf){
                    //std::cout << "Cannot move link because " << IsEnd << " , " << IsFull << std::endl;
                    TimeAwareHeapInsert(eventindex, RateMove[eventindex-1],systime,tstep); // TEMP
                } else{ // Move the link
                    //std::cout << "Link moving! " << std::endl;
                    //std::cout << "Old head and tail " << _LinkHeads[iLink] << " , " <<  _LinkTails[iLink] << std::endl;
                    if (IsTail){
                        _LinkTails[iLink]++;
                    } else {
                        _LinkHeads[iLink]++;
                    }
                    _TotalNumberBound[PtToMove]--;
                    _TotalNumberBound[PtToMove+1]++;
                    // Recompute force
                    vec ParallelForces = ComputeParForce(iLink,uniPts,uniTanVecs,RealShifts);
                    double RateMove_i = std::max(_UnloadedRate*(1+ParallelForces[0]/_FStall),0.0); // moving velocity
                    double RateMove_j = std::max(_UnloadedRate*(1+ParallelForces[1]/_FStall),0.0); // moving velocity
                    RateMove[2*iLink]=RateMove_i;
                    RateMove[2*iLink+1]=RateMove_j;
                    // Add to heap
                    TimeAwareHeapInsert(2*iLink+1, RateMove_i,systime,tstep);
                    TimeAwareHeapInsert(2*iLink+2, RateMove_j,systime,tstep);
                    //std::cout << "New head and tail " << _LinkHeads[iLink] << " , " <<  _LinkTails[iLink] << std::endl;
                }
            } else { // Moving unloaded links
                int SiteToMove = eventindex - UnloadedStart;
                bool IsEnd = ((SiteToMove+1) % _NSitesPerFib)==0;
		    bool IsFull = false;
		    if (!IsEnd){
                    IsFull = _TotalNumberBound[SiteToMove+1]==_MaxNumberPerSite;
		    }
		    if (IsEnd || IsFull){
                    //std::cout << "Cannot move link because " << IsEnd << " , " << IsFull << std::endl;
                    TimeAwareHeapInsert(eventindex,_UnloadedRate*_FreeLinkBound[SiteToMove],systime,tstep); // TEMP
                } else{ // Move the link
                    _TotalNumberBound[SiteToMove]--;
                    _FreeLinkBound[SiteToMove]--;
                    _TotalNumberBound[SiteToMove+1]++;   
                    _FreeLinkBound[SiteToMove+1]++;
                    TimeAwareHeapInsert(UnloadedStart+SiteToMove, _UnloadedRate*_FreeLinkBound[SiteToMove],systime,tstep);
                    TimeAwareHeapInsert(UnloadedStart+SiteToMove+1, _UnloadedRate*_FreeLinkBound[SiteToMove+1],systime,tstep);
                }
            }
            topOfHeap(eventindex,systime);
        }
    }
        

    void deleteLinksFromSites(npInt pySiteNumbers){
        // Traverse the lists in reverse order
        intvec SitesToDelete(pySiteNumbers.size());
        std::memcpy(SitesToDelete.data(),pySiteNumbers.data(),pySiteNumbers.size()*sizeof(int));
        for (int iLink=_nDoubleBoundLinks-1; iLink >=0; iLink--){
            // Unbind it (remove from lists)
            for (uint iSite = 0; iSite < SitesToDelete.size(); iSite++){
                if (_LinkHeads[iLink]==SitesToDelete[iSite] || _LinkTails[iLink]==SitesToDelete[iSite]){
                    _TotalNumberBound[_LinkHeads[iLink]]--;
                    _TotalNumberBound[_LinkTails[iLink]]--;
                    deleteLink(iLink);
                    break; // out of loop over sites
                }
            }
        } // end loop over existing links
        // Remove singly bound links as well
        for (uint iSite = 0; iSite < SitesToDelete.size(); iSite++){
            _FreeLinkBound[SitesToDelete[iSite]]=0;
            _TotalNumberBound[SitesToDelete[iSite]] = 0;
        }
    }

    npInt getNBoundEnds(){
        // Copy _FreeLinkBound to numpy array
        // return 1-D NumPy array
        // allocate py::array (to pass the result of the C++ function to Python)
        auto pyNBoundEnds = py::array_t<int>(_TotSites);
        auto result_buffer = pyNBoundEnds.request();
        int *result_ptr    = (int *) result_buffer.ptr;
        // copy std::vector -> py::array
        std::memcpy(result_ptr,_FreeLinkBound.data(),_TotSites*sizeof(int));
        return pyNBoundEnds;
    }

    npInt getLinkHeadsOrTails(bool Head){
        // Copy heads or tails to numpy array
        // return 1-D NumPy array
        // allocate py::array (to pass the result of the C++ function to Python)
        auto pyArray = py::array_t<int>(_nDoubleBoundLinks);
        auto result_buffer = pyArray.request();
        int *result_ptr    = (int *) result_buffer.ptr;
        // copy std::vector -> py::array
        if (Head){
            std::memcpy(result_ptr,_LinkHeads.data(),_nDoubleBoundLinks*sizeof(int));
        } else {
            std::memcpy(result_ptr,_LinkTails.data(),_nDoubleBoundLinks*sizeof(int));
        }
        return pyArray;
    }

    npDoub getLinkShifts(){
        // Copy _LinkShifts to numpy array (2D)

        ssize_t              ndim    = 2;
        std::vector<ssize_t> shape   = { _nDoubleBoundLinks , 3 };
        std::vector<ssize_t> strides = { sizeof(double)*3 , sizeof(double) };

        // return 2-D NumPy array
        return py::array(py::buffer_info(
            _LinkShiftsPrime.data(),                       /* data as contiguous array  */
            sizeof(double),                          /* size of one scalar        */
            py::format_descriptor<double>::format(), /* data type                 */
            ndim,                                    /* number of dimensions      */
            shape,                                   /* shape of the matrix       */
            strides                                  /* strides for each axis     */
        ));
    }

    void setLinks(npInt pyHeads, npInt pyTails, npDoub pyShifts, npInt pyFreelyBoundPerSite){
        /*
        Set the link heads, tails, shifts, and freely bound (singly bound) links at each site from
        python arrays
        */
        int nLinksIn = pyHeads.size();
        if (nLinksIn  > _maxLinks){
            _maxLinks = 2*nLinksIn;
            _LinkHeads.resize(_maxLinks);
            _LinkTails.resize(_maxLinks);
            _LinkShiftsPrime.resize(3*_maxLinks);
        }
        _nDoubleBoundLinks = nLinksIn;
        std::cout << "Setting C++ links with " << _nDoubleBoundLinks << " links " << std::endl;
        std::memcpy(_LinkHeads.data(),pyHeads.data(),pyHeads.size()*sizeof(int));
        std::memcpy(_LinkTails.data(),pyTails.data(),pyTails.size()*sizeof(int));
        std::memcpy(_LinkShiftsPrime.data(),pyShifts.data(),pyShifts.size()*sizeof(double));
        std::memcpy(_FreeLinkBound.data(),pyFreelyBoundPerSite.data(),pyFreelyBoundPerSite.size()*sizeof(int));
        for (int iSite = 0; iSite < _TotSites; iSite++){
            _TotalNumberBound[iSite] = _FreeLinkBound[iSite];
        }
        for (int iLink = 0; iLink < _nDoubleBoundLinks; iLink++){
            _TotalNumberBound[_LinkHeads[iLink]]++;
            _TotalNumberBound[_LinkTails[iLink]]++;
        }
    }

    private:
        int _TotSites, _nDoubleBoundLinks, _maxLinks, _MaxNumberPerSite, _NSitesPerFib;
        double _kon, _konSecond, _koff, _koffSecond;// rates
        double _kT, _restlen, _KStiffness; // for distance-dependent binding
        double _upperCLBound, _lowerCLBound; // absolute distances
        double _UnloadedRate, _FStall, _SinFreq; // motors
        intvec _FreeLinkBound, _LinkHeads, _LinkTails, _TotalNumberBound;
        vec _LinkShiftsPrime, _RealDistances;
        std::uniform_real_distribution<double> unif;
        std::mt19937_64 rng;
        bool _LockContract, _SpatialBinding;

        double logrand(){
            return -log(1.0-unif(rng));
        }
        
        double LinkDistance(int iPt, int jPt, const vec &uniPts, int iPair, const vec &RealShift, double &xBar){
            vec3 displacement;
            for (int d=0; d < 3; d++){
                displacement[d]=uniPts[3*iPt+d]-uniPts[3*jPt+d]-RealShift[3*iPair+d];
            }
            xBar = 0.5*(uniPts[3*iPt]+uniPts[3*jPt]+RealShift[3*iPair]);
            return normalize(displacement);
        }
            
        double BindingRate(double r){
            if (_kT == 0.0){
                return 1.0*_konSecond;
            } 
            double Energy = 0.5*_KStiffness*(r-_restlen)*(r-_restlen);
            return _konSecond*exp(-Energy/_kT);
        }
        
        double UnbindingRate(double r){
            return _koffSecond;
            /*if (_kT == 0.0){
                return 1.0*_koffSecond;
            } 
            double Energy = 0.5*_KStiffness*(r-_restlen)*(r-_restlen);
            return _koffSecond*exp(0.25*Energy/_kT);*/
        }
            

        void updateSecondBindingRate(int BoundEnd, const intvec &FirstLinkBySite, const intvec &NextLinkByLink, 
            const vec &ProposedDistances, const vec &XBars, double systime, double tstep){
            /*
            Update the binding rate of every link with left end = BoundEnd.
            SecondEndRates = new rates for binding the second end, startPair = first index of sorted pairs of links that
            has left end BoundEnd.
            */
            int CurLink = FirstLinkBySite[BoundEnd];
            while (CurLink!=-1){
                int DistancesIndex = CurLink/2;
                double BaseRate = BindingRate(ProposedDistances[DistancesIndex]);
                if (_SpatialBinding){
                    BaseRate*=abs(sin(_SinFreq*XBars[DistancesIndex]));
                }
                TimeAwareHeapInsert(CurLink+1, BaseRate*_FreeLinkBound[BoundEnd],systime,tstep);
                // notice we start indexing from 1, this is how fortran is written
                CurLink = NextLinkByLink[CurLink];
            }
        }

        void TimeAwareHeapInsert(int index, double rate, double systime,double timestep){
            /*
            Insert in heap if rate is nonzero and the generated time is less than the time step
            */
            if (rate==0){
                deleteFromHeap(index); // delete any previous copy
                return;
            } else if (rate < 0){
                std::cout << "Your rate is less than zero!" << std::endl;
                throw std::runtime_error("Your rate is less than zero!");
            }
            double newtime = systime+logrand()/rate;
            //std::cout << "Trying to insert index " << index << " with time " << newtime << " (systime = " << systime << std::endl;
            if (newtime < timestep){
                //std::cout << "Inserting " << index << " with time " << newtime << std::endl;
                insertInHeap(index,newtime);
            } else {
                deleteFromHeap(index);
            }
        }
        
        vec ComputeParForce(const int &iLink, const vec &uniPoints, const vec &uniTanVecs, const vec &Shifts){
            vec ParForces(2);
            int iPt = _LinkHeads[iLink];
            int jPt = _LinkTails[iLink];
            // Displacement vector
            vec3 ds, tau_i, tau_j;
            for (int d =0; d < 3; d++){
                ds[d] = uniPoints[3*iPt+d] -  uniPoints[3*jPt+d] - Shifts[3*iLink+d];
                tau_i[d] = uniTanVecs[3*iPt+d];
                tau_j[d] = uniTanVecs[3*jPt+d];
            }
            double nds = normalize(ds);
            normalize(tau_i);
            normalize(tau_j);
            double dsDottau_i = dot(ds,tau_i);
            double dsDottau_j = dot(ds,tau_j);
            ParForces[0] = -_KStiffness*(nds-_restlen)*dsDottau_i;
            ParForces[1] = _KStiffness*(nds-_restlen)*dsDottau_j;
            return ParForces;
        }

        void deleteLink(int linkNum){
            // Unbind it (remove from lists)
            _LinkHeads[linkNum] = _LinkHeads[_nDoubleBoundLinks-1];
            _LinkTails[linkNum] = _LinkTails[_nDoubleBoundLinks-1];
            _RealDistances[linkNum] = _RealDistances[_nDoubleBoundLinks-1];
            for (int d=0; d < 3; d++){
                _LinkShiftsPrime[3*linkNum+d] = _LinkShiftsPrime[3*(_nDoubleBoundLinks-1)+d];
            }
            _nDoubleBoundLinks-=1;
        }
        
        npDoub make1DPyArray(vec &cppvec){
            // Return a 1D py array
            // allocate py::array (to pass the result of the C++ function to Python)
            auto pyArray = py::array_t<double>(cppvec.size());
            auto result_buffer = pyArray.request();
            double *result_ptr    = (double *) result_buffer.ptr;
            // copy std::vector -> py::array
            std::memcpy(result_ptr,cppvec.data(),cppvec.size()*sizeof(double));
            return pyArray;
        }


};


PYBIND11_MODULE(EndedCrossLinkedNetwork, m) {
    py::class_<EndedCrossLinkedNetwork>(m, "EndedCrossLinkedNetwork")
        .def(py::init<int, int, vec, vec, double, double, double, double>())
        .def("updateNetwork", &EndedCrossLinkedNetwork::updateNetwork)
        .def("WalkLinks",&EndedCrossLinkedNetwork::WalkLinks)
        .def("MotorSpeeds", &EndedCrossLinkedNetwork::MotorSpeeds)
        .def("ChangeLockContract",&EndedCrossLinkedNetwork::ChangeLockContract)
        .def("ChangeSpatialBinding",&EndedCrossLinkedNetwork::ChangeSpatialBinding)
        .def("SetMotorParams",&EndedCrossLinkedNetwork::SetMotorParams)
        .def("getNBoundEnds", &EndedCrossLinkedNetwork::getNBoundEnds)
        .def("getLinkHeadsOrTails",&EndedCrossLinkedNetwork::getLinkHeadsOrTails)
        .def("getLinkShifts", &EndedCrossLinkedNetwork::getLinkShifts)
        .def("setLinks",&EndedCrossLinkedNetwork::setLinks)
        .def("deleteLinksFromSites", &EndedCrossLinkedNetwork::deleteLinksFromSites);
}
