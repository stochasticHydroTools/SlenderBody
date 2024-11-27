#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <chrono>
#include "DomainC.cpp"
#include "types.h"
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

    EndedCrossLinkedNetwork(int TotSites, int maxLinksPerSite, vec rates, vec3 DomLengths, vec CLBounds,double kT,
        double restlen, double KStiffness,double CLseed):_Dom(DomLengths){
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
        _UnbindingRates = vec(_maxLinks,0);
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

     }
     
     void SetMotorParams(double UnloadedRate, double StallForce, int SitesPerFib){
        _UnloadedRate=UnloadedRate;
        _FStall = StallForce;
        _NSitesPerFib = SitesPerFib;
     }

     ~EndedCrossLinkedNetwork(){
        deleteHeap();
     }


     void updateNetwork(double tstep, npInt pyMaybeBindingPairs, npDoub pyuniPts, double g){
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

        // Determine pairs that can actually bind
        int nPotentialLinks = MaybeBindingPairs.size()/2;
        vec PrimedShifts(6*nPotentialLinks,0);
        intvec BindingPairs(4*nPotentialLinks,-1);
        vec distances(2*nPotentialLinks,0.0);
        int nTruePairs = EliminateLinksOutsideRange(MaybeBindingPairs, uniPts, g, PrimedShifts, BindingPairs,distances);
        //std::cout << "Number of true possible pairs " << nTruePairs/2 << std::endl;
        resetHeap();

        // Compute the binding rates of those pairs and sort list of pairs
        intvec numLinksPerSite(_TotSites,0);
        for (int iPair=0; iPair < nTruePairs; iPair++){
            int BoundEnd = BindingPairs[2*iPair]; // they are stacked so the bound end comes last
            numLinksPerSite[BoundEnd]++;
            //std::cout << "CL pair " << BindingPairs[2*iPair] << " , " << BindingPairs[2*iPair+1] << std::endl;
        }
        // Sort the possible links
        intvec startPair(_TotSites,0);
        for (int iSite=1; iSite < _TotSites; iSite++){ // cumulative sum
            startPair[iSite]=startPair[iSite-1]+numLinksPerSite[iSite-1];
        }
        intvec SortedLinks(2*nTruePairs,-1);
        vec BaseSortedRates(nTruePairs,0), SortedShifts(3*nTruePairs,0);
        for (int iPair=0; iPair < nTruePairs; iPair++){
            int BoundEnd = BindingPairs[2*iPair];
            int index = startPair[BoundEnd];
            int isOccupied = SortedLinks[2*index];
            while (isOccupied > -1){
                index++;
                isOccupied = SortedLinks[2*index];
            } // is occupied is -1 at end of loop; can insert there
            SortedLinks[2*index] = BoundEnd;
            SortedLinks[2*index+1] = BindingPairs[2*iPair+1]; // unbound end
            for (int d=0; d < 3; d++){
                SortedShifts[3*index+d] = PrimedShifts[3*iPair+d];
            }
            if (_kT == 0.0){
                BaseSortedRates[index] = _konSecond*1.0; // add distance dependent factor
            } else {
                //std::cout << "Distance dependent binding!" << std::endl;
                double Energy = 0.5*_KStiffness*(distances[iPair]-_restlen)*(distances[iPair]-_restlen);
                BaseSortedRates[index] = _konSecond*exp(-Energy/_kT);
            }
            TimeAwareHeapInsert(index+1, BaseSortedRates[index]*_FreeLinkBound[BoundEnd],0,tstep);// notice we start indexing from 1, this is how fortran is written
        }

        // Single binding to a site
        double RateFreeBind = _kon*_TotSites;
        int indexFreeBinding = nTruePairs+1;
        TimeAwareHeapInsert(indexFreeBinding, RateFreeBind,0,tstep);

        // Single unbinding from a site
        vec RatesFreeUnbind(_TotSites);
        int indexFreeUnbinding = indexFreeBinding+1;
        for (int iSite = 0; iSite < _TotSites; iSite++){
            RatesFreeUnbind[iSite] = _koff*_FreeLinkBound[iSite];
            TimeAwareHeapInsert(indexFreeUnbinding+iSite, RatesFreeUnbind[iSite],0,tstep);
        }

        // One end of CL unbinding
        int indexSecondUnbind = indexFreeUnbinding+_TotSites;
        for (int iLink=0; iLink < _nDoubleBoundLinks; iLink++){
            _UnbindingRates[iLink]=2*_koffSecond;
            TimeAwareHeapInsert(indexSecondUnbind+iLink,_UnbindingRates[iLink],0,tstep);
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
                TimeAwareHeapInsert(eventindex,_UnbindingRates[linkNum],systime,tstep);
                TimeAwareHeapInsert(indexSecondUnbind+_nDoubleBoundLinks,0,systime,tstep); // remove last one
                _TotalNumberBound[UnboundEnd]--; // Only the unbound end loses a link
                // Add a free end at the other site. Always assume the remaining bound end is at the left
                //std::cout << "Index " << eventindex << "link num " << linkNum << " sites " << BoundEnd << " , " << UnboundEnd <<
                //    ", CL end unbind at time " << systime << std::endl;
                _FreeLinkBound[BoundEnd]++;
            } else { // CL binding
                // The index now determines which pair of sites the CL is binding to
                // In addition, we always assume the bound end is the one at the left
                int pairToBind = eventindex-1;
                //std::cout << "Index " << eventindex << ", CL link " << pairToBind << " both ends bind at time " << systime << std::endl;
                BoundEnd = SortedLinks[2*pairToBind];
                UnboundEnd = SortedLinks[2*pairToBind+1];
                // Link can only bind if the unbound end is available
                if (_TotalNumberBound[UnboundEnd] < _MaxNumberPerSite){
                    _LinkHeads[_nDoubleBoundLinks] = BoundEnd;
                    _LinkTails[_nDoubleBoundLinks] = UnboundEnd;
                    _UnbindingRates[_nDoubleBoundLinks] = 2*_koffSecond;
                    TimeAwareHeapInsert(indexSecondUnbind+_nDoubleBoundLinks,_UnbindingRates[_nDoubleBoundLinks],systime,tstep);
                     // unbound end picks up a link, bound end loses a free link
                    _TotalNumberBound[UnboundEnd]++;
                    _FreeLinkBound[BoundEnd]--; // one less free link bound
                    for (int d=0; d < 3; d++){
                        _LinkShiftsPrime[3*_nDoubleBoundLinks+d] = SortedShifts[3*pairToBind+d];
                    }
                    _nDoubleBoundLinks++;
                } // otherwise nothing happens
            }
            if (_nDoubleBoundLinks == _maxLinks){ // double size of link arrays if necessary
                _maxLinks*=2;
                _LinkHeads.resize(_maxLinks);
                _LinkTails.resize(_maxLinks);
                _UnbindingRates.resize(_maxLinks);
                _LinkShiftsPrime.resize(3*_maxLinks);
                std::cout << "Expanding array size to " << _maxLinks << std::endl;
            }
            // Rates of CL binding change based on number of bound ends
            updateSecondBindingRate(BoundEnd, BaseSortedRates, startPair[BoundEnd], numLinksPerSite[BoundEnd], systime, tstep);

            // Update unbinding event at BoundEnd (the end whose state has changed)
            RatesFreeUnbind[BoundEnd] = _koff*_FreeLinkBound[BoundEnd];
            //std::cout << "About to insert unbinding single end in heap with index " << BoundEnd+indexFreeUnbinding <<  std::endl;
            TimeAwareHeapInsert(BoundEnd+indexFreeUnbinding,RatesFreeUnbind[BoundEnd],systime,tstep);

            topOfHeap(eventindex,systime);
            //std::cout << "[" << eventindex << " , " << systime << "]" << std::endl;
            // Debugging check
            intvec TotNum2(_TotSites, 0);
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
            }
        }
    }
    
    void WalkLinks(double tstep, npDoub pyuniPts, npDoub pyuniTanVecs, npDoub pyShifts){       
        // TODO: Compile and test this simple method
        // Not yet figured out: what happens when a link is sitting on the end. It 
        // can't move, but could it unbind? Need to figure that out (shouldn't be too 
        // big a deal to take it off)
        // Convert numpy to C++ vector and reset heap for the beginning of the step
        vec uniPts(pyuniPts.size());
        std::memcpy(uniPts.data(),pyuniPts.data(),pyuniPts.size()*sizeof(double));
        vec uniTanVecs(pyuniTanVecs.size());
        std::memcpy(uniTanVecs.data(),pyuniTanVecs.data(),pyuniTanVecs.size()*sizeof(double));
        vec Shifts(pyShifts.size());
        std::memcpy(Shifts.data(),pyShifts.data(),pyShifts.size()*sizeof(double));
        resetHeap();

        // Walking of doubly-bound links (finite force)
        vec RateMove(2*_nDoubleBoundLinks); // might not be needed
        for (int iLink=0; iLink < _nDoubleBoundLinks; iLink++){
            // Compute the force
            int iPt = _LinkHeads[iLink];
            int jPt = _LinkTails[iLink];
            vec ParallelForces = ComputeParForce(iLink,iPt,jPt,uniPts,uniTanVecs,Shifts);
            double RateMove_i = std::max(_UnloadedRate*(1+ParallelForces[0]/_FStall),0.0); // moving velocity
            double RateMove_j = std::max(_UnloadedRate*(1+ParallelForces[1]/_FStall),0.0); // moving velocity
            RateMove[2*iLink]=RateMove_i;
            RateMove[2*iLink+1]=RateMove_j;
            std::cout << "Rates of movement link " << iLink << " = " << RateMove[2*iLink] << " , " << RateMove[2*iLink+1] << std::endl;
            // Add to heap
            TimeAwareHeapInsert(2*iLink+1, RateMove_i,0,tstep);
            TimeAwareHeapInsert(2*iLink+2, RateMove_j,0,tstep);
        }
        
        // Walking of singly-bound link
        int UnloadedStart = 2*_nDoubleBoundLinks+1;
        vec RatesFreeMove(_TotSites);
        for (int iSite = 0; iSite < _TotSites; iSite++){
            RatesFreeMove[iSite] = _UnloadedRate*_FreeLinkBound[iSite];
            TimeAwareHeapInsert(UnloadedStart+iSite, RatesFreeMove[iSite],0,tstep);
        }

        double systime;
        int eventindex;
        std::cout << "Try to find top of heap " << std::endl;
        topOfHeap(eventindex,systime);
        //std::cout << "Top of heap is at index " << eventindex << " and time " << systime << std::endl;
        while (eventindex > 0) {
            std::cout << "System time " << systime << std::endl;
            std::cout << "Event index " << eventindex << std::endl;
            if (eventindex < UnloadedStart){ // Moving loaded links
                int iLink = (eventindex-1)/2; 
                bool IsTail = (eventindex % 2)==0;
                std::cout << "Index (starts from 1!) " << eventindex << std::endl;
                int PtToMove = _LinkHeads[iLink];
                if (IsTail){
                    PtToMove = _LinkTails[iLink];
                }
                std::cout << "PtToMove " << PtToMove << std::endl;
                // Check that it can move to the next site
                bool IsEnd = ((PtToMove+1) % _NSitesPerFib)==0;
                bool IsFull = _TotalNumberBound[PtToMove+1]==_MaxNumberPerSite;
                if (IsEnd || IsFull){
                    std::cout << "Cannot move link because " << IsEnd << " , " << IsFull << std::endl;
                    TimeAwareHeapInsert(eventindex, RateMove[eventindex-1],systime,tstep); // TEMP
                } else{ // Move the link
                    std::cout << "Link moving! " << std::endl;
                    std::cout << "Old head and tail " << _LinkHeads[iLink] << " , " <<  _LinkTails[iLink] << std::endl;
                    if (IsTail){
                        _LinkTails[iLink]++;
                    } else {
                        _LinkHeads[iLink]++;
                    }
                    _TotalNumberBound[PtToMove]--;
                    _TotalNumberBound[PtToMove+1]++;
                    // Recompute force
                    vec ParallelForces = ComputeParForce(iLink,_LinkHeads[iLink],_LinkTails[iLink],uniPts,uniTanVecs,Shifts);
                    double RateMove_i = std::max(_UnloadedRate*(1+ParallelForces[0]/_FStall),0.0); // moving velocity
                    double RateMove_j = std::max(_UnloadedRate*(1+ParallelForces[1]/_FStall),0.0); // moving velocity
                    RateMove[2*iLink]=RateMove_i;
                    RateMove[2*iLink+1]=RateMove_j;
                    // Add to heap
                    TimeAwareHeapInsert(2*iLink+1, RateMove_i,systime,tstep);
                    TimeAwareHeapInsert(2*iLink+2, RateMove_j,systime,tstep);
                    std::cout << "New head and tail " << _LinkHeads[iLink] << " , " <<  _LinkTails[iLink] << std::endl;
                }
            } else { // Moving unloaded links
                int SiteToMove = eventindex - UnloadedStart;
                bool IsEnd = ((SiteToMove+1) % _NSitesPerFib)==0;
                bool IsFull = _TotalNumberBound[SiteToMove+1]==_MaxNumberPerSite;
                if (IsEnd || IsFull){
                    std::cout << "Cannot move link because " << IsEnd << " , " << IsFull << std::endl;
                    TimeAwareHeapInsert(eventindex, RatesFreeMove[SiteToMove],systime,tstep); // TEMP
                } else{ // Move the link
                    _TotalNumberBound[SiteToMove]--;
                    _FreeLinkBound[SiteToMove]--;
                    _TotalNumberBound[SiteToMove+1]++;   
                    _FreeLinkBound[SiteToMove+1]++;
                    // Update the movement rate at these sites
                    RatesFreeMove[SiteToMove] = _UnloadedRate*_FreeLinkBound[SiteToMove];
                    RatesFreeMove[SiteToMove+1] = _UnloadedRate*_FreeLinkBound[SiteToMove+1];
                    TimeAwareHeapInsert(UnloadedStart+SiteToMove, RatesFreeMove[SiteToMove],systime,tstep);
                    TimeAwareHeapInsert(UnloadedStart+SiteToMove+1, RatesFreeMove[SiteToMove+1],systime,tstep);
                }
            }
            topOfHeap(eventindex,systime);
            intvec TotNum2(_TotSites, 0);
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
                    eventindex=-1;
                    std::cout << "Tot number bound at site " << iSite << " = " << _TotalNumberBound[iSite] << std::endl;
                    std::cout << "Compared to " << iSite << " = " << TotNum2[iSite] << std::endl;
                }
                if (_TotalNumberBound[iSite] > _MaxNumberPerSite){
                    std::cout << "EXCEED MAX!" << std::endl;
                }
            }
        }
        std::cout << "END" << std::endl;
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
        double _UnloadedRate, _FStall; // motors
        intvec _FreeLinkBound, _LinkHeads, _LinkTails, _TotalNumberBound;
        DomainC _Dom;
        vec _LinkShiftsPrime, _UnbindingRates;
        std::uniform_real_distribution<double> unif;
        std::mt19937_64 rng;

        double logrand(){
            return -log(1.0-unif(rng));
        }

        void updateSecondBindingRate(int BoundEnd, vec &BaseSortedRates, int startPair, int numLinks,double systime, double tstep){
            /*
            Update the binding rate of every link with left end = BoundEnd.
            SecondEndRates = new rates for binding the second end, startPair = first index of sorted pairs of links that
            has left end BoundEnd.
            */
            for (int ThisLink = startPair; ThisLink < startPair+numLinks; ThisLink++){
                //std::cout << "About to insert CL second bind in heap with index " << ThisLink+1 << std::endl;
                TimeAwareHeapInsert(ThisLink+1, BaseSortedRates[ThisLink]*_FreeLinkBound[BoundEnd],systime,tstep);
            }
        }

        void TimeAwareHeapInsert(int index, double rate, double systime,double timestep){
            /*
            Insert in heap if rate is nonzero and the generated time is less than the time step
            */
            if (rate==0){
                deleteFromHeap(index); // delete any previous copy
                return;
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
        
        vec ComputeParForce(const int &iLink, const int &iPt,const int &jPt,const vec &uniPoints,
            const vec &uniTanVecs, const vec &Shifts){
            vec ParForces(2);
            // Displacement vector
            vec3 ds, tau_i, tau_j;
            for (int d =0; d < 3; d++){
                ds[d] = uniPoints[3*iPt+d] -  uniPoints[3*jPt+d] - Shifts[3*iLink+d];
                tau_i[d] = uniTanVecs[3*iPt+d];
                tau_j[d] = uniTanVecs[3*jPt+d];
            }
            std::cout << "The tangent vector 1 is " << tau_i[0] << " , " << tau_i[1] << " , " << tau_i[2] << std::endl;
            std::cout << "The tangent vector 2 is " << tau_j[0] << " , " << tau_j[1] << " , " << tau_j[2] << std::endl;
            std::cout << "ds is " << ds[0] << " , " << ds[1] << " , " << ds[2] << std::endl;
            double nds = normalize(ds);
            normalize(tau_i);
            normalize(tau_j);
            double dsDottau_i = dot(ds,tau_i);
            double dsDottau_j = dot(ds,tau_j);
            ParForces[0] = -_KStiffness*(nds-_restlen)*dsDottau_i;
            ParForces[1] = _KStiffness*(nds-_restlen)*dsDottau_j;
            std::cout << "Force mag = " << _KStiffness*(nds-_restlen) << std::endl;
            std::cout << "Force mag i = " << ParForces[0] << std::endl;
            std::cout << "Force mag j = " << ParForces[1] << std::endl;
            return ParForces;
        }

        int EliminateLinksOutsideRange(const intvec &newLinkSites, const vec &uniPts, double g, vec &PrimedShifts, intvec &ActualPossibleLinks, vec &distances){
            /*
            From the list of potential links (newLinkSites) and the uniform points (uniPts), and the strain in the coordinate system, calculate
            the actual displacement vector of each link (including the relevant shift from the array PrimedShifits, and add it
            to the array ActualPossibleLinks
            */
            int nPotentialLinks = newLinkSites.size()/2;
            int nLPoss=0;
            for (int iMaybeLink=0; iMaybeLink < nPotentialLinks; iMaybeLink++){
                int iPt = newLinkSites[2*iMaybeLink];
                int jPt = newLinkSites[2*iMaybeLink+1];
                vec3 rvec;
                for (int d=0; d < 3; d++){
                    rvec[d] = uniPts[3*iPt+d]-uniPts[3*jPt+d];
                }
                //std::cout << "i " << uniPts[3*iPt] << " , " << uniPts[3*iPt+1] << " , " << uniPts[3*iPt+2] << std::endl;
                //std::cout << "j " << uniPts[3*jPt] << " , " << uniPts[3*jPt+1] << " , " << uniPts[3*jPt+2] << std::endl;
                //std::cout << "rvec " << rvec[0] << " , " << rvec[1] << " , " << rvec[2] << std::endl;
                _Dom.calcShifted(rvec,g);
                //std::cout << "shift rvec " << rvec[0] << " , " << rvec[1] << " , " << rvec[2] << std::endl;
                double r = sqrt(dot(rvec,rvec));
                //std::cout << "Actual r " << r << std::endl;
                //std::cout << "Upper and lower bound " << _upperCLBound << " , " << _lowerCLBound << std::endl;
                if (r < _upperCLBound && r > _lowerCLBound){
                    distances[nLPoss]=r;
                    distances[nLPoss+1]=r;
                    ActualPossibleLinks[2*nLPoss]=iPt;
                    ActualPossibleLinks[2*nLPoss+1]=jPt;
                    ActualPossibleLinks[2*nLPoss+2]=jPt;
                    ActualPossibleLinks[2*nLPoss+3]=iPt;
                    vec3 PrimeShift;
                    for (int d =0; d < 3; d++){
                        PrimeShift[d] = uniPts[3*iPt+d]-uniPts[3*jPt+d]-rvec[d];
                    }
                    _Dom.PrimeCoords(PrimeShift,g);
                    for (int d =0; d < 3; d++){
                        PrimedShifts[3*nLPoss+d] = PrimeShift[d];
                        PrimedShifts[3*(nLPoss+1)+d] = -PrimeShift[d];
                    }
                    nLPoss+=2;
                }
            }
            PrimedShifts.resize(3*nLPoss);
            ActualPossibleLinks.resize(2*nLPoss);
            distances.resize(nLPoss);
            return nLPoss;
        }

        void deleteLink(int linkNum){
            // Unbind it (remove from lists)
            _LinkHeads[linkNum] = _LinkHeads[_nDoubleBoundLinks-1];
            _LinkTails[linkNum] = _LinkTails[_nDoubleBoundLinks-1];
            _UnbindingRates[linkNum] = _UnbindingRates[_nDoubleBoundLinks-1];
            for (int d=0; d < 3; d++){
                _LinkShiftsPrime[3*linkNum+d] = _LinkShiftsPrime[3*(_nDoubleBoundLinks-1)+d];
            }
            _nDoubleBoundLinks-=1;
        }


};


PYBIND11_MODULE(EndedCrossLinkedNetwork, m) {
    py::class_<EndedCrossLinkedNetwork>(m, "EndedCrossLinkedNetwork")
        .def(py::init<int, int, vec, vec3, vec, double, double, double, double>())
        .def("updateNetwork", &EndedCrossLinkedNetwork::updateNetwork)
        .def("WalkLinks",&EndedCrossLinkedNetwork::WalkLinks)
        .def("SetMotorParams",&EndedCrossLinkedNetwork::SetMotorParams)
        .def("getNBoundEnds", &EndedCrossLinkedNetwork::getNBoundEnds)
        .def("getLinkHeadsOrTails",&EndedCrossLinkedNetwork::getLinkHeadsOrTails)
        .def("getLinkShifts", &EndedCrossLinkedNetwork::getLinkShifts)
        .def("setLinks",&EndedCrossLinkedNetwork::setLinks)
        .def("deleteLinksFromSites", &EndedCrossLinkedNetwork::deleteLinksFromSites);
}
