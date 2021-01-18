#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <chrono>
#include "Domain.cpp"
#include "types.h"

/**
C++ class for a cross-linked network where we track each end separately
This class is bound to python using pybind11
See the file pyEndedCrossLinkers.cpp for the bindings. 

There are 3 reactions
1) Binding of a floating link to one site (rate _kon)
2) Unbinding of a link that is bound to one site to become free (reverse of 1, rate _koff)
3) Binding of a singly-bound link to another site to make a doubly-bound CL (rate _konSecond)
4) Unbinding of a double bound link in one site to make a single-bound CL (rate _koffSecond)
5) Binding of both ends of a CL (rate _kDoubleOn)
6) Unbinding of both ends of a CL (rate _kDoubleOff)
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
       
    EndedCrossLinkedNetwork(int TotSites, vec rates, double CLseed){
        _TotSites = TotSites; 
        _kon = rates[0];
        _konSecond = rates[1];
        _koff = rates[2];
        _koffSecond = rates[3];
        _kDoubleOn = rates[4];
        _kDoubleOff = rates[5];
        _FreeLinkBound = intvec(TotSites,0);
        int maxLinks = std::max(2*int(_konSecond/_koffSecond*_kon/_koff*_TotSites),100); // guess for max # of links
        _LinkHeads = intvec(maxLinks,-1);
        _LinkTails = intvec(maxLinks,-1);
        initializeHeap(maxLinks+_TotSites+4);
        rng.seed(CLseed);
        unif = std::uniform_real_distribution<double>(0.0,1.0);
        _nFreeEnds =0;
        _nDoubleBoundLinks = 0;
        
     }
     
     ~EndedCrossLinkedNetwork(){
        deleteHeap();
     }
     
     void updateNetwork(double tstep, npInt pyBindingPairs){
        /*
        Update the network using Kinetic MC.
        Inputs: tstep = time step. pyBindingPairs = 2D numpy array, where the first
        column is the bound end and the second column is the unbound end, of possible
        link binding site pairs. 
        */
        // Convert numpy to C++ vector and reset heap for the beginning of the step
        intvec BindingPairs(pyBindingPairs.size());
        std::memcpy(BindingPairs.data(),pyBindingPairs.data(),pyBindingPairs.size()*sizeof(int));
        int nTruePairs = BindingPairs.size()/2;
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
        vec RatesSecondBind(nTruePairs); 
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
            RatesSecondBind[index] = _konSecond*_FreeLinkBound[BoundEnd];
            TimeAwareHeapInsert(index+1, RatesSecondBind[index],0,tstep); // notice we start indexing from 1, this is how fortran is written
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
        double RateSecondUnbind = 2*_koffSecond*_nDoubleBoundLinks;
        int indexSecondUnbind = indexFreeUnbinding+_TotSites;
        TimeAwareHeapInsert(indexSecondUnbind,RateSecondUnbind,0,tstep);
        
        // Events for double binding and unbinding
        double RateDoubleBind = _kDoubleOn*nTruePairs;
        int indexDoubleBind = indexSecondUnbind+1;
        TimeAwareHeapInsert(indexDoubleBind,RateDoubleBind,0,tstep);
        
        double RateDoubleUnbind = _kDoubleOff*_nDoubleBoundLinks;
        int indexDoubleUnbind = indexSecondUnbind+2;
        TimeAwareHeapInsert(indexDoubleUnbind,RateDoubleUnbind,0,tstep);
        
        double systime;
        int eventindex, BoundEnd, UnboundEnd, PlusOrMinusSingleBound;
        topOfHeap(eventindex,systime);
        //std::cout << "Top of heap is at index " << eventindex << " and time " << systime << std::endl;
        while (eventindex > 0) {
            bool linkChange = false;
            if (eventindex==indexFreeBinding){ // end binding, choose random site and bind an end
                //std::cout << "Index " << eventindex << ", single binding at time " << systime << std::endl;
                BoundEnd = int(unif(rng)*_TotSites);
                PlusOrMinusSingleBound=1;
                TimeAwareHeapInsert(indexFreeBinding,RateFreeBind,systime,tstep);
            } else if (eventindex >= indexFreeUnbinding && eventindex < indexSecondUnbind){ // single end unbind
                BoundEnd = eventindex-indexFreeUnbinding;
                //std::cout << "Index " << eventindex << ", single unbinding at time " << systime << std::endl;
                PlusOrMinusSingleBound=-1;
            } else if (eventindex == indexSecondUnbind || eventindex==indexDoubleUnbind){ // CL unbinding
                int linkNum = int(unif(rng)*_nDoubleBoundLinks);
                BoundEnd = _LinkHeads[linkNum];
                if (unif(rng) < 0.5){
                    BoundEnd = _LinkTails[linkNum];    
                }
                // Unbind it (remove from lists)
                _LinkHeads[linkNum] = _LinkHeads[_nDoubleBoundLinks-1];
                _LinkTails[linkNum] = _LinkTails[_nDoubleBoundLinks-1];
                _nDoubleBoundLinks-=1;
                linkChange = true;
                if (eventindex == indexSecondUnbind){
                    // Add a free end at the other site. Always assume the remaining bound end is at the left
                    //std::cout << "Index " << eventindex << ", CL end unbind at time " << systime << std::endl;
                    PlusOrMinusSingleBound=1;
                } else{
                    //std::cout << "Index " << eventindex << ", CL both ends unbind at time " << systime << std::endl;
                    PlusOrMinusSingleBound=0;
                }
            } else { // CL binding
                // The index now determines which pair of sites the CL is binding to
                // In addition, we always assume the bound end is the one at the left 
                int pairToBind;
                if (eventindex == indexDoubleBind){
                    //std::cout << "Index " << eventindex << ", CL both ends bind at time " << systime << std::endl;
                    pairToBind = int(unif(rng)*nTruePairs);
                    PlusOrMinusSingleBound=0;
                    TimeAwareHeapInsert(indexDoubleBind,RateDoubleBind,systime,tstep);
                } else{
                    pairToBind = eventindex-1;
                    //std::cout << "Index " << eventindex << ", CL link " << linkIndex << " both ends bind at time " << systime << std::endl;
                    PlusOrMinusSingleBound=-1;
                }
                BoundEnd = SortedLinks[2*pairToBind];
                UnboundEnd = SortedLinks[2*pairToBind+1];
                _LinkHeads[_nDoubleBoundLinks] = BoundEnd;
                _LinkTails[_nDoubleBoundLinks] = UnboundEnd;
                _nDoubleBoundLinks+=1;
                linkChange = true;
            }   
            // Recompute subset of rates and times
            _nFreeEnds+= PlusOrMinusSingleBound;
            _FreeLinkBound[BoundEnd]+= PlusOrMinusSingleBound;
            
            // Rates of CL binding change based on number of bound ends
            updateSecondBindingRate(BoundEnd, RatesSecondBind, startPair[BoundEnd], numLinksPerSite[BoundEnd], systime, tstep);
             
            // Update unbinding event at BoundEnd (the end whose state has changed)
            RatesFreeUnbind[BoundEnd] = _koff*_FreeLinkBound[BoundEnd];
            //std::cout << "About to insert unbinding single end in heap with index " << BoundEnd+indexFreeUnbinding << 
            //    " and new time " << TimesFreeUnbind[BoundEnd] << std::endl;
            TimeAwareHeapInsert(BoundEnd+indexFreeUnbinding,RatesFreeUnbind[BoundEnd],systime,tstep);     
            
            // Update unbinding events (links)
            if (linkChange){
                RateSecondUnbind = 2*_koffSecond*_nDoubleBoundLinks;
                TimeAwareHeapInsert(indexSecondUnbind,RateSecondUnbind,systime,tstep);
                RateDoubleUnbind = _kDoubleOff*_nDoubleBoundLinks;
                TimeAwareHeapInsert(indexDoubleUnbind,RateDoubleUnbind,systime,tstep);
            }
            topOfHeap(eventindex,systime);
            //std::cout << "Top of heap is at index " << eventindex << " and time " << systime << std::endl;
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
                     
    private:
        int _TotSites, _nFreeEnds, _nDoubleBoundLinks;
        double _kon, _konSecond, _koff, _koffSecond, _kDoubleOn, _kDoubleOff; // rates
        intvec _FreeLinkBound, _LinkHeads, _LinkTails;
        std::uniform_real_distribution<double> unif;
        std::mt19937_64 rng;
        
        double logrand(){
            return -log(1.0-unif(rng));
        }
        
        void updateSecondBindingRate(int BoundEnd, vec &SecondEndRates, int startPair, int numLinks,double systime, double tstep){
            /*
            Update the binding rate of every link with left end = BoundEnd
            */
            for (int ThisLink = startPair; ThisLink < startPair+numLinks; ThisLink++){
                SecondEndRates[ThisLink] = _konSecond*_FreeLinkBound[BoundEnd];
                TimeAwareHeapInsert(ThisLink+1, SecondEndRates[ThisLink],systime,tstep);
            }
        }

        void TimeAwareHeapInsert(int index, double rate, double systime,double timestep){
            if (rate==0){
                deleteFromHeap(index); // delete any previous copy
                return;
            }
            double newtime = systime+logrand()/rate;
            if (newtime < timestep){
                //std::cout << "Inserting " << index << " with time " << eventtime << std::endl;
                insertInHeap(index,newtime);
            } else {
                deleteFromHeap(index);
            }
        }
 
};    
