module MinHeapModule
   use iso_c_binding
   implicit none
   public
   
   ! Constants that require recompilation:
   integer, parameter, private :: dp=c_double, sp=c_float
   integer, parameter, public :: wp = dp ! Double precision event times
   !integer, parameter, public :: wp = sp ! Single precision event times

   ! Element of the priority queue.
   type, public, bind(C) :: EventType
      real (wp) :: time
      integer(c_int) :: elementIndex = 0 ! An integer telling you id of this event (a hash into any data structure that contains further info on events)
        ! Important: This Fortran code starts indexing from 1 not from 0!
        ! Important: elementIndex==0 means invalid entry in this implementation, so if you want to start indexing with zero 
   end type EventType

   type, private :: MinHeap

      integer(c_int) :: heapSize = 0 ! Current heap size
      type (EventType), allocatable :: priorityQueue (:)
      integer(c_int), allocatable :: positionInHeap (:) ! Important for event-driven simulation, same size/indexing as priorityQueue
   
   end type MinHeap
   
   ! To avoid issues with interfacing with C/python, we make the heap a global private variable
   ! so there can only be ONE heap per run in this version of the code, sorry
   ! It is possible to avoid using C_F_POINTER but for now I don't do that
   type(MinHeap), private :: heap
   private :: modifyHeap

contains
 
   ! void initializeHeap(int size)
   subroutine initializeHeap(size) bind(C,name="initializeHeap")
      !type (MinHeap), intent(inout) :: heap
      integer, intent(in), value :: size

      if(allocated(heap%priorityQueue)) then ! Re-initialize heap
         deallocate(heap%priorityQueue, heap%positionInHeap)
      end if
      allocate(heap%priorityQueue(size+1)) ! We use size+1 to avoid overflow when referring to the child of some element
      allocate(heap%positionInHeap(size))
      call resetHeap()

   end subroutine initializeHeap

   ! void deleteHeap()
   subroutine deleteHeap() bind(C,name="deleteHeap")
      !type (MinHeap), intent(inout) :: heap

      if(allocated(heap%priorityQueue)) then ! Re-initialize heap
         deallocate(heap%priorityQueue, heap%positionInHeap)
      end if
      heap%heapSize=0

   end subroutine deleteHeap
   
   ! void increaseHeapSize()
   subroutine increaseHeapSize() bind(C,name="increaseHeapSize")
      ! Doubles the size of the heap, used when we run out of space
      ! We do not provide a routine to shrink, if you want that just call initializeHeap again with new size
      !type (MinHeap), intent(inout) :: heap
   
      type (EventType), allocatable :: priorityQueue_new (:)
      integer, allocatable :: positionInHeap_new (:)
      integer :: new_size
      
      if(.not.allocated(heap%priorityQueue)) stop "Trying to resize un-initialized heap"
      
      new_size=2*size(heap%positionInHeap)
      print *, "HEAP: Increasing size of heap from ", size(heap%priorityQueue), " to ", new_size
      allocate(priorityQueue_new(new_size+1), positionInHeap_new(new_size))
      priorityQueue_new(:)%time = huge(1.0_wp) ! Times of events for each element relative to the last time the queue was reset (time origin)
      priorityQueue_new(:)%elementIndex = 0
      positionInHeap_new = 0
      
      priorityQueue_new(1:heap%heapSize)=heap%priorityQueue(1:heap%heapSize)
      positionInHeap_new(1:size(heap%positionInHeap))=heap%positionInHeap(1:size(heap%positionInHeap))
      
      ! Reallocate arrays:
      call move_alloc(priorityQueue_new, heap%priorityQueue)
      call move_alloc(positionInHeap_new, heap%positionInHeap)
      
   end subroutine increaseHeapSize

   ! void resetHeap()
   subroutine resetHeap() bind(C,name="resetHeap")
      !type (MinHeap), intent(inout) :: heap

      ! In our specific case this is the beginning of the time step

      heap%priorityQueue(:)%time = huge(1.0_wp) ! Times of events for each element relative to the last time the queue was reset (time origin)
      heap%priorityQueue(:)%elementIndex = 0
      heap%heapSize = 0
      heap%positionInHeap = 0
      
   end subroutine resetHeap

   ! void topOfHeap(int *index, float/double *time)
   subroutine topOfHeap(elementIndex,time) bind(C,name="topOfHeap")
      !type (MinHeap), intent(inout) :: heap
      integer, intent(out) :: elementIndex
      real (wp), intent(out) :: time
      
      elementIndex = heap%priorityQueue(1)%elementIndex
      time = heap%priorityQueue(1)%time
   
   end subroutine
      
   ! void insertInHeap(int index, float/double time)
   subroutine insertInHeap(elementIndex,time) bind(C,name="insertInHeap") ! AD: Made this routine safe to call even if already in heap
      !type (MinHeap), intent(inout) :: heap
      integer, intent(in), value :: elementIndex
      real (wp), intent(in), value :: time

      integer :: parent, child, tempElementIndex
      real (wp) :: tempTime
      
      do while (elementIndex > size(heap%positionInHeap)) 
        call increaseHeapSize() 
      end do
      
      if(heap%positionInHeap(elementIndex)/=0) then ! This is already in the heap
         call  updateHeap(elementIndex,time)
         return
      end if

      heap%heapSize = heap%heapSize+1
      heap%priorityQueue(heap%heapSize)%time = time
      heap%priorityQueue(heap%heapSize)%elementIndex = elementIndex
      ! QY: Fortran ignores fractional part for integer division so I didn't distinguish between odd and even cases.   
      parent = heap%heapSize/2
      child = heap%heapSize
      heap%positionInHeap(elementIndex) = heap%heapSize
      SiftUpLoop: do
         if (parent==0) exit SiftUploop
         if (heap%priorityQueue(parent)%time < heap%priorityQueue(child)%time) exit SiftUpLoop
         tempTime = heap%priorityQueue(child)%time 
         tempelementIndex = heap%priorityQueue(child)%elementIndex
         heap%priorityQueue(child)%time = heap%priorityQueue(parent)%time
         heap%priorityQueue(child)%elementIndex = heap%priorityQueue(parent)%elementIndex
         heap%priorityQueue(parent)%time = tempTime
         heap%priorityQueue(parent)%elementIndex = tempelementIndex
         heap%positionInHeap(heap%priorityQueue(child)%elementIndex)=child
         heap%positionInHeap(heap%priorityQueue(parent)%elementIndex)=parent
         child = parent
         parent = parent/2
      end do SiftUpLoop
   end subroutine insertInHeap

   ! void deleteFromHeap(int index)
   subroutine deleteFromHeap(elementIndex) bind(C,name="deleteFromHeap") 
      ! AD made this routine safe to call on previously deleted entries
      !type (MinHeap), intent(inout) :: heap
      integer, intent(in), value :: elementIndex

      integer :: tempIndex, parent, self, last
      real (wp) :: tempTime

      if(elementIndex > size(heap%positionInHeap)) then
        return
      end if
      if(heap%positionInHeap(elementIndex)==0) return ! Not in queue!
      if (heap%heapSize==0) then
         write (*,*) elementIndex
         write (*,*) heap%priorityQueue(1)%time,heap%priorityQueue(2)%time,heap%priorityQueue(3)%time
         stop "Deleting from an empty heap"
      end if
      tempIndex = heap%positionInHeap(elementIndex)

      heap%positionInHeap(elementIndex)=0
      if (tempIndex/=heap%heapSize) then
         last = heap%priorityQueue(heap%heapSize)%elementIndex
         heap%priorityQueue(tempIndex)%time = heap%priorityQueue(heap%heapSize)%time
         heap%priorityQueue(tempIndex)%elementIndex = last
         heap%positionInHeap(last)=tempIndex
      end if
      heap%priorityQueue(heap%heapSize)%time = huge(1.0_wp)
      heap%priorityQueue(heap%heapSize)%elementIndex = 0
      heap%heapSize = heap%heapSize - 1

      if ((heap%heapSize>0).and.(tempIndex<=heap%heapSize)) then
         if(heap%priorityQueue(tempIndex)%elementIndex==0) then
            write (*,*) "Heap size is:", heap%heapSize, "cell index is:", elementIndex, "original last one is:", last 
            stop "Deleting heap value of an element that is not in the heap"
         end if
         call modifyHeap(heap,tempIndex)
         if(heap%priorityQueue(tempIndex)%elementIndex==0) stop "Problem arising after modifyHeap"
      end if
   end subroutine deleteFromHeap

   ! void updateHeap(int index, float/double time)
   subroutine updateHeap(elementIndex,time) bind(C,name="updateHeap")
      !type (MinHeap), intent(inout) :: heap
      integer, intent(in), value :: elementIndex
      real (wp), intent(in), value :: time

      integer :: tempIndex

      tempIndex = heap%positionInHeap(elementIndex)
      if(tempIndex==0) stop "Trying to update heap value that is not in the heap"
      ! QY: we replace the time by a new one. There's no need to replace the elementIndex since it's the same element.
      heap%priorityQueue(tempIndex)%time = time
      call modifyHeap(heap,tempIndex)

   end subroutine updateHeap
   
   ! void testHeap()
   subroutine testHeap() bind(C,name="testHeap")
      ! type (MinHeap), intent(inout) :: heap

      integer :: i, leftChild, rightChild

      do i = 1, heap%heapSize
         leftChild = i*2
         rightChild = i*2+1
         if (leftChild<=heap%heapSize) then
            if (heap%priorityQueue(leftChild)%time < heap%priorityQueue(i)%time) stop 'Wrong heap'
         end if
         if (rightChild<=heap%heapSize) then
            if (heap%priorityQueue(rightChild)%time < heap%priorityQueue(i)%time) stop 'Wrong heap'
         end if
      end do
      do i = 1, heap%heapSize
         if (heap%positionInHeap(heap%priorityQueue(i)%elementIndex)/=i) stop 'Problem in heap'
      end do
   end subroutine testHeap

   ! -------------------------------------------------
   ! Private routines
   ! -------------------------------------------------

   ! QY: new subroutine, used in delete and renew
   subroutine modifyHeap(heap,tempIndex)
      type (MinHeap), intent(inout) :: heap
      integer, intent(in) :: tempIndex

      integer :: parent, leftChild, rightChild, self, minimum
      integer :: tempelementIndex, elementIndex
      real (wp) :: tempTime

      parent = tempIndex/2
      self = tempIndex
      leftChild = tempIndex*2
      tempTime = heap%priorityQueue(tempIndex)%time
      elementIndex = heap%priorityQueue(tempIndex)%elementIndex
      if(elementIndex==0) stop "Trying to modify value that is not actually in the heap"
      
      tempelementIndex = elementIndex
      SiftUpLoop: do
         if (parent==0) exit SiftUpLoop
         if (heap%priorityQueue(parent)%time < tempTime) exit SiftUpLoop
         heap%priorityQueue(self)%time = heap%priorityQueue(parent)%time
         heap%priorityQueue(self)%elementIndex = heap%priorityQueue(parent)%elementIndex
         heap%priorityQueue(parent)%time = tempTime
         heap%priorityQueue(parent)%elementIndex = tempelementIndex
         heap%positionInHeap(heap%priorityQueue(parent)%elementIndex)=parent
         heap%positionInHeap(heap%priorityQueue(self)%elementIndex)=self
         self = parent
         parent = parent/2
      end do SiftUpLoop

      ! In case the node is not sifted up at all, then we may need to sift it down.
      if (self == tempIndex) then
         SiftDownLoop: do
            if (leftChild > heap%heapSize) exit SiftDownLoop
            rightChild=leftChild+1
            minimum = self
            if (heap%priorityQueue(leftChild)%time < tempTime) minimum = leftChild
            if (heap%priorityQueue(rightChild)%time < heap%priorityQueue(minimum)%time) minimum = rightChild
            if (minimum == self) exit SiftDownLoop
            if (minimum == leftChild) then
               tempelementIndex = heap%priorityQueue(self)%elementIndex
               heap%priorityQueue(self)%time = heap%priorityQueue(leftChild)%time
               heap%priorityQueue(self)%elementIndex = heap%priorityQueue(leftChild)%elementIndex
               heap%priorityQueue(leftChild)%time = tempTime
               heap%priorityQueue(leftChild)%elementIndex = tempelementIndex
               heap%positionInHeap(heap%priorityQueue(leftChild)%elementIndex)=leftChild
               heap%positionInHeap(heap%priorityQueue(self)%elementIndex)=self
               self = leftChild
               leftChild = self*2
            else
               tempelementIndex = heap%priorityQueue(self)%elementIndex
               heap%priorityQueue(self)%time = heap%priorityQueue(rightChild)%time
               heap%priorityQueue(self)%elementIndex = heap%priorityQueue(rightChild)%elementIndex
               heap%priorityQueue(rightChild)%time = tempTime
               heap%priorityQueue(rightChild)%elementIndex = tempelementIndex
               heap%positionInHeap(heap%priorityQueue(rightChild)%elementIndex)=rightChild
               heap%positionInHeap(heap%priorityQueue(self)%elementIndex)=self
               self = rightChild
               leftChild = self*2
            end if

         end do SiftDownLoop
      end if
      heap%positionInHeap(elementIndex)=self

   end subroutine modifyHeap

end module MinHeapModule
