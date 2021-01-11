import heapq
import numpy as np

class EventQueue(object):
    
    def __init__(self,EventList,nEventsMax):
        self._que = EventList;
        self._indexInQueue = np.zeros(nEventsMax,dtype=np.int64)-1;
        self.heapifyque();
        
    def heapifyque(self):
        """Transform list into a heap, in-place, in O(len(x)) time."""
        n = len(self._que)
        # Transform bottom-up.  The largest index there's any point to looking at
        # is the largest with a child index in-range, so must have 2*i + 1 < n,
        # or i < (n-1)/2.  If n is even = 2*j, this is (2*j-1)/2 = j-1/2 so
        # j-1 is the largest, which is n//2 - 1.  If n is odd = 2*j+1, this is
        # (2*j+1-1)/2 = j so j-1 is the largest, and that's again n//2-1.
        for i in reversed(range(n)):
            self.siftup(i)
    
    def __str__(self):
        for i in self._que:
            print(i)
        return ''
    
    def heappush(self,index,time,maxTime):
        """Push item onto heap, maintaining the heap invariant."""
        item = CrossLinkingEvent(time,index)
        if (item.time() <= maxTime):
            self._que.append(item)
            self.siftdown(0, len(self._que)-1)

    def heappop(self):
        """Pop the smallest item off the heap, maintaining the heap invariant."""
        lastelt = self._que.pop()    # raises appropriate IndexError if heap is empty
        if self._que:
            returnitem = self._que[0]
            self._indexInQueue[returnitem.index()]=-1;
            self._que[0] = lastelt
            self.siftup(0)
            return returnitem
        else:
            self._indexInQueue[lastelt.index()]=-1;
        return lastelt
    
    #def heappop(self):
    #    return self._que[0];
    
    def heapupdate(self,index,newtime,tstep):
        pos = self._indexInQueue[index];
        if (pos==-1): # item already popped, or not in list yet 
            self.heappush(index,newtime,tstep);
        else:
            heapitem = self._que[pos]
            oldtime = heapitem.time();
            heapitem.setNewTime(newtime);
            if (newtime > oldtime):
                self.siftup(pos);
            else:
                self.siftdown(0,pos);
    
    def updateEventIndex(self,oldIndex,newIndex):
        pos = self._indexInQueue[oldIndex];
        if (pos > -1): # otherwise do nothing
            Event = self._que[pos];
            Event.setNewIndex(newIndex);
            self._indexInQueue[newIndex] = pos;
            self._indexInQueue[oldIndex] = -1;   
    
    def checkIndices(self,maxindex):
        for i in range(len(self._que)):
            eventindex = self._que[i].index();
            if (eventindex > maxindex):
                print('There is an event with index %d when the max inde is %d' %(eventindex,maxindex))
                raise ValueError('Index is too large!')
            if not (i==self._indexInQueue[eventindex]):
                print('Mismatcched index %d' %self._indexInQueue[eventindex])
                print('The index you are looking for is %d' %i)
                print('The index of the event is %d' %eventindex)
                raise ValueError('Bug in value list!')      
            
    # 'heap' is a heap at all indices >= startpos, except possibly for pos.  pos
    # is the index of a leaf with a possibly out-of-order value.  Restore the
    # heap invariant.
    def siftdown(self,startpos, pos):
        newitem = self._que[pos]
        # Follow the path to the root, moving parents down until finding a place
        # newitem fits.
        while pos > startpos:
            parentpos = (pos - 1) >> 1
            parent = self._que[parentpos]
            if newitem < parent:
                self._que[pos] = parent
                self._indexInQueue[parent.index()] = pos;
                pos = parentpos
                continue
            break
        self._que[pos] = newitem
        self._indexInQueue[newitem.index()] = pos;
    
    
    def siftup(self, pos):
        endpos = len(self._que)
        startpos = pos
        newitem = self._que[pos]
        # Bubble up the smaller child until hitting a leaf.
        childpos = 2*pos + 1    # leftmost child position
        while childpos < endpos:
            # Set childpos to index of smaller child.
            rightpos = childpos + 1
            if rightpos < endpos and not self._que[childpos] < self._que[rightpos]:
                childpos = rightpos
            # Move the smaller child up.
            self._que[pos] = self._que[childpos]
            self._indexInQueue[self._que[pos].index()] = pos;
            pos = childpos
            childpos = 2*pos + 1
        # The leaf at pos is empty now.  Put newitem there, and bubble it up
        # to its final resting place (by sifting its parents down).
        self._que[pos] = newitem
        self._indexInQueue[newitem.index()] = pos;
        self.siftdown(startpos, pos)
    
class CrossLinkingEvent(object):
    
    def __init__(self,TimeItHappens,EventIndex):
        
        self._TimeItHappens = TimeItHappens;
        self._EventIndex = EventIndex;
    
    def index(self):
        return self._EventIndex;
    
    def time(self):
        return self._TimeItHappens;
    
    def setNewTime(self,time):
        self._TimeItHappens = time;
    
    def setNewIndex(self,newind):
        self._EventIndex = newind;
    
    def __str__(self):
        return '[%f, %d]' %(self._TimeItHappens,self._EventIndex)
        
    def __gt__(self,other):
        return self._TimeItHappens > other._TimeItHappens;
    
    def __lt__(self,other):
        return self._TimeItHappens < other._TimeItHappens;
    
    def __ge__(self,other):
        return self._TimeItHappens >= other._TimeItHappens;
    
    def __le__(self,other):
        return self._TimeItHappens <= other._TimeItHappens;
        
    def __eq__(self,other):
        return self._TimeItHappens == other._TimeItHappens;
