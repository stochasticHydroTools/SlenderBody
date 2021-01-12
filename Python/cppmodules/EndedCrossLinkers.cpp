#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <chrono>
#include "Domain.cpp"
#include "types.h"

/**
This is a set of C++ functions that are being used
for cross linker related calculations in the C++ code
This version is for cross linkers with unique ends
**/

extern "C"{

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
        int maxLinks = std::max(2*int(_konSecond/_koffSecond*_kon/_koff*_TotSites),10);
        _LinkHeads = intvec(maxLinks,-1);
        _LinkTails = intvec(maxLinks,-1);
        initializeHeap(maxLinks+_TotSites+4);
        rng.seed(CLseed);
        unif = std::uniform_real_distribution<double>(0.0,1.0);
        _nFreeEnds =0;
        _nDoubleBoundLinks = 0;
        
     }
     
     
     void updateNetwork(double tstep, npInt pyBindingPairs){
        /*
        Update the network using Kinetic MC.
        Inputs: fiberCol = fiberCollection object of the fibers,
        Dom = Domain object, tstep = the timestep we are updating the network
        by (a different name than dt)
        This is an event-driven algorithm. See comments throughout, but
        the basic idea is to sample times for each event, then update 
        those times as the state of the network changes. 
        */
        // Convert numpy to stdvector
        intvec BindingPairs(pyBindingPairs.size());
        std::memcpy(BindingPairs.data(),pyBindingPairs.data(),pyBindingPairs.size()*sizeof(int));
        int nTruePairs = BindingPairs.size()/2;
        
        
        // Compute the binding rates of those pairs
        vec RatesSecondBind(nTruePairs), TimesSecondBind(nTruePairs);
        for (int iPair=0; iPair < nTruePairs; iPair++){
            int BoundEnd = BindingPairs[2*iPair]; // they are stacked so the bound end comes last
            RatesSecondBind[iPair] = _konSecond*_FreeLinkBound[BoundEnd];
            TimesSecondBind[iPair] = logrand()/RatesSecondBind[iPair];
            //std::cout << "CL pair " << BindingPairs[2*iPair] << " , " << BindingPairs[2*iPair+1] << std::endl;
            //std::cout << " Time " << TimesSecondBind[iPair] << " and index " << iPair << std::endl;
            insertInHeap(iPair+1, TimesSecondBind[iPair]);
        }
        // Make linked lists
        intvec FirstLinkPerSite(_TotSites,-1);
        intvec NextLinkByLink(nTruePairs,-1);
        makePairLinkLists(nTruePairs,BindingPairs,FirstLinkPerSite,NextLinkByLink);
        
        // Single binding to a site
        double RateFreeBind = _kon*_TotSites;
        double TimeFreeBind = logrand()/RateFreeBind;
        int indexFreeBinding = nTruePairs+1;
        //std::cout << "Binding time " << TimeFreeBind << " and index " << indexFreeBinding << std::endl;
        insertInHeap(indexFreeBinding, TimeFreeBind);
        
        // Single unbinding from a site
        vec RatesFreeUnbind(_TotSites);
        vec TimesFreeUnbind(_TotSites);
        int indexFreeUnbinding = indexFreeBinding+1;
        for (int iSite = 0; iSite < _TotSites; iSite++){
            RatesFreeUnbind[iSite] = _koff*_FreeLinkBound[iSite];
            TimesFreeUnbind[iSite] = logrand()/RatesFreeUnbind[iSite];
            insertInHeap(indexFreeUnbinding+iSite, TimesFreeUnbind[iSite]);
            //std::cout << "Unbind from site " << iSite << " time " << TimesFreeUnbind[iSite] << " and index " << indexFreeUnbinding+iSite << std::endl;
        }
        
        // One end of CL unbinding
        double RateSecondUnbind = _koff*_nDoubleBoundLinks;
        double TimeSecondUnbind = logrand()/RateSecondUnbind;
        int indexSecondUnbind = indexFreeUnbinding+_TotSites;
        //std::cout << "One end CL unbind time " << TimeSecondUnbind << " and index " << indexSecondUnbind << std::endl;
        insertInHeap(indexSecondUnbind,TimeSecondUnbind);
        
        // Events for double binding and unbinding
        double RateDoubleBind = _kDoubleOn*nTruePairs;
        double TimeDoubleBind = logrand()/RateDoubleBind;
        int indexDoubleBind = indexSecondUnbind+1;
        insertInHeap(indexDoubleBind,TimeDoubleBind);
        //std::cout << "Two end CL bind time " << TimeDoubleBind << " and index " << indexDoubleBind << std::endl;
        
        double RateDoubleUnbind = _kDoubleOff*_nDoubleBoundLinks;
        double TimeDoubleUnbind = logrand()/RateDoubleUnbind;
        int indexDoubleUnbind = indexSecondUnbind+2;
        insertInHeap(indexDoubleUnbind,TimeDoubleUnbind);
        //std::cout << "Two end CL unbind time " << TimeDoubleUnbind << " and index " << indexDoubleUnbind << std::endl;
        
        double systime;
        int eventindex, BoundEnd, UnboundEnd, PlusOrMinusSingleBound;
        topOfHeap(eventindex,systime);
        while (systime < tstep) {
            //std::cout << "Top of heap is at index " << eventindex << " and time " << systime << std::endl;
            bool linkChange = false;
            if (eventindex==indexFreeBinding){ // end binding, choose random site and bind an end
                systime = TimeFreeBind;
                //std::cout << "Index " << eventindex << ", single binding at time " << systime << std::endl;
                BoundEnd = int(unif(rng)*_TotSites);
                PlusOrMinusSingleBound=1;
                TimeFreeBind = logrand()/RateFreeBind+systime;
                insertInHeap(indexFreeBinding,TimeFreeBind);
            } else if (eventindex >= indexFreeUnbinding && eventindex < indexSecondUnbind){ // single end unbind
                BoundEnd = eventindex-indexFreeUnbinding;
                systime = TimesFreeUnbind[BoundEnd];
                //std::cout << "Index " << eventindex << ", single unbinding at time " << systime << std::endl;
                PlusOrMinusSingleBound=-1;
            } else if (eventindex == indexSecondUnbind || eventindex==indexDoubleUnbind){ // CL unbinding
                int linkNum = int(unif(rng)*_nDoubleBoundLinks);
                BoundEnd = _LinkHeads[linkNum];
                // Unbind it (remove from lists)
                _LinkHeads[linkNum] = _LinkHeads[_nDoubleBoundLinks-1];
                _LinkTails[linkNum] = _LinkTails[_nDoubleBoundLinks-1];
                _nDoubleBoundLinks-=1;
                linkChange = true;
                if (eventindex == indexSecondUnbind){
                    // Add a free end at the other site. Always assume the remaining bound end is at the left
                    systime = TimeSecondUnbind;
                    //std::cout << "Index " << eventindex << ", CL end unbind at time " << systime << std::endl;
                    PlusOrMinusSingleBound=1;
                } else{
                    systime = TimeDoubleUnbind;
                    //std::cout << "Index " << eventindex << ", CL both ends unbind at time " << systime << std::endl;
                    PlusOrMinusSingleBound=0;
                }
            } else { // CL binding
                // The index now determines which pair of sites the CL is binding to
                // In addition, we always assume the bound end is the one at the left 
                if (eventindex == indexDoubleBind){
                    systime = TimeDoubleBind;
                    //std::cout << "Index " << eventindex << ", CL both ends bind at time " << systime << std::endl;
                    int pair = int(unif(rng)*nTruePairs);
                    //std::cout << "The chosen pair " << pair << std::endl;
                    BoundEnd = BindingPairs[2*pair];
                    UnboundEnd = BindingPairs[2*pair+1]; 
                    PlusOrMinusSingleBound=0;
                    TimeDoubleBind = logrand()/RateDoubleBind+systime;
                    //std::cout << "About to insert in heap with new time " << TimeDoubleBind << std::endl;
                    insertInHeap(indexDoubleBind,TimeDoubleBind);
                    //std::cout << "Done with heap insert " << std::endl;
                } else{
                    int linkIndex = eventindex-1;
                    systime = TimesSecondBind[linkIndex];
                    //std::cout << "Index " << eventindex << ", CL link " << linkIndex << " both ends bind at time " << systime << std::endl;
                    BoundEnd = BindingPairs[2*linkIndex];
                    UnboundEnd = BindingPairs[2*linkIndex+1];
                    PlusOrMinusSingleBound=-1;
                }
                //std::cout << "About to update tails and heads " << std::endl;
                _LinkHeads[_nDoubleBoundLinks] = BoundEnd;
                _LinkTails[_nDoubleBoundLinks] = UnboundEnd;
                _nDoubleBoundLinks+=1;
                linkChange = true;
            }   
            //std::cout << "Made it to the end " << std::endl;;
            // Recompute subset of rates and times
            _nFreeEnds+= PlusOrMinusSingleBound;
            _FreeLinkBound[BoundEnd]+= PlusOrMinusSingleBound;
            
            // Rates of CL binding change based on number of bound ends
            updateSecondBindingRate(BoundEnd, RatesSecondBind, TimesSecondBind, FirstLinkPerSite, NextLinkByLink, systime);
            //std::cout << "Finished second binding update " << std::endl;;
             
            // Update unbinding event at BoundEnd (the end whose state has changed)
            RatesFreeUnbind[BoundEnd] = _koff*_FreeLinkBound[BoundEnd];
            TimesFreeUnbind[BoundEnd] = logrand()/RatesFreeUnbind[BoundEnd]+systime;
            insertInHeap(BoundEnd+indexFreeUnbinding,TimesFreeUnbind[BoundEnd]);     
            
            // Update unbinding events (links)
            if (linkChange){
                RateSecondUnbind = _koff*_nDoubleBoundLinks;
                TimeSecondUnbind = logrand()/RateSecondUnbind+systime;
                insertInHeap(indexSecondUnbind,TimeSecondUnbind);
                RateDoubleUnbind = _kDoubleOff*_nDoubleBoundLinks;
                TimeDoubleUnbind = logrand()/RateDoubleUnbind+systime;
                insertInHeap(indexDoubleUnbind,TimeDoubleUnbind);
            }
            topOfHeap(eventindex,systime);
        }
        //deleteHeap();
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
        // Copy _FreeLinkBound to numpy array
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
           
    //void setName(const std::string &name_) { name = name_; }
    //const std::string &getName() const { return name; }
    //const double &getFirstGen() const { return numkids[0]; }
    //void pysetNumChildren(py::array_t<double> pynumkids){
        // Convert to C++ array
       // vec kids(pynumkids.size());
        //std::memcpy(kids.data(),pynumkids.data(),pynumkids.size()*sizeof(double));
        //setNumChildren(kids);
    //}     
    double getkoff(){return _koff;}
        
    private:
        int _TotSites, _nFreeEnds, _nDoubleBoundLinks;
        double _kon, _konSecond, _koff, _koffSecond, _kDoubleOn, _kDoubleOff; // rates
        intvec _FreeLinkBound, _LinkHeads, _LinkTails;
        std::uniform_real_distribution<double> unif;
        std::mt19937_64 rng;
        
        double logrand(){
            return -log(1.0-unif(rng));
        }
        
        void updateSecondBindingRate(int BoundEnd, vec &SecondEndRates, vec &SecondEndBindingTimes,
            const intvec &FirstLink,const intvec &NextLink,double systime){
            int ThisLink = FirstLink[BoundEnd];
            while (ThisLink > -1){
                SecondEndRates[ThisLink] = _konSecond*_FreeLinkBound[BoundEnd];
                double eltime = logrand()/SecondEndRates[ThisLink];
                SecondEndBindingTimes[ThisLink] = eltime + systime;
                //std::cout << "Inserting " << ThisLink << " as index " <<ThisLink+1 << " with time " << SecondEndBindingTimes[ThisLink] << std::endl;
                insertInHeap(ThisLink+1, SecondEndBindingTimes[ThisLink]);
                ThisLink = NextLink[ThisLink];
            }
        }
        
        void makePairLinkLists(int nTruePairs,const intvec &BindingPairs, intvec &FirstLinkPerSite, intvec &NextLinkByLink){
            for (int iPair=0; iPair < nTruePairs; iPair++){
                int BoundEnd = BindingPairs[2*iPair];     
                if (FirstLinkPerSite[BoundEnd]==-1){
                    FirstLinkPerSite[BoundEnd] = iPair;
                } else {
                    int prevnewLink;
                    int newLink = FirstLinkPerSite[BoundEnd]; 
                    while (newLink > -1){
                        prevnewLink = newLink;
                        newLink = NextLinkByLink[newLink];   
                    } 
                    NextLinkByLink[prevnewLink] = iPair;
                } // endif
            } // end for
            /*for (int iEnd=0; iEnd < _TotSites; iEnd++){
                std::cout << "Site " << iEnd << " first link " << FirstLinkPerSite[iEnd] << std::endl;
            }
            for (int iLink=0; iLink < nTruePairs; iLink++){
                std::cout << "Link " << iLink << " next link " << NextLinkByLink[iLink] << std::endl;
            }    */
        } // end makePairLinkLists
 
};    
