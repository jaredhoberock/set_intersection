#pragma once

#include <thrust/iterator/iterator_traits.h>
#include <thrust/pair.h>
#include <thrust/detail/minmax.h>

namespace detail
{

template<bool UpperBound, typename IntT, typename It, typename T, typename Comp>
__host__ __device__ void BinarySearchIteration(It data, int& begin, int& end,
	T key, int shift, Comp comp) {

	IntT scale = (1<< shift) - 1;
	int mid = (int)((begin + scale * end)>> shift);

	T key2 = data[mid];
	bool pred = UpperBound ? !comp(key, key2) : comp(key2, key);
	if(pred) begin = (int)mid + 1;
	else end = mid;
}

template<bool UpperBound, typename T, typename It, typename Comp>
__host__ __device__ int BinarySearch(It data, int count, T key, Comp comp) {
	int begin = 0;
	int end = count;
	while(begin < end) 
		BinarySearchIteration<UpperBound, int>(data, begin, end, key, 1, comp);
	return begin;
}

template<bool UpperBound, typename IntT, typename T, typename It, typename Comp>
__host__ __device__ int BiasedBinarySearch(It data, int count, T key, 
	IntT levels, Comp comp) {
	int begin = 0;
	int end = count;

	if(levels >= 4 && begin < end)
		BinarySearchIteration<UpperBound, IntT>(data, begin, end, key, 9, comp);
	if(levels >= 3 && begin < end)
		BinarySearchIteration<UpperBound, IntT>(data, begin, end, key, 7, comp);
	if(levels >= 2 && begin < end)
		BinarySearchIteration<UpperBound, IntT>(data, begin, end, key, 5, comp);
	if(levels >= 1 && begin < end)
		BinarySearchIteration<UpperBound, IntT>(data, begin, end, key, 4, comp);

	while(begin < end)
		BinarySearchIteration<UpperBound, IntT>(data, begin, end, key, 1, comp);
	return begin;
}

template<bool UpperBound, typename It1, typename It2, typename Comp>
__host__ __device__ int MergePath(It1 a, int aCount, It2 b, int bCount, 
	int diag, Comp comp) {

	typedef typename thrust::iterator_traits<It1>::value_type T;

	int begin = thrust::max(0, diag - bCount);
	int end   = thrust::min(diag, aCount);

	while(begin < end) {
		int mid = (begin + end)>> 1;
		T aKey = a[mid];
		T bKey = b[diag - 1 - mid];
		bool pred = UpperBound ? comp(aKey, bKey) : !comp(bKey, aKey);
		if(pred) begin = mid + 1;
		else end = mid;
	}
	return begin;
}


// Locate the merge path search intervals for blocked merges. This is used for 
// merging many lists recursively in merge sort. coop is the number of CTAs
// cooperating in producing each sorted sequence. coop must be a power of 2 
// starting at 2. blockSize is the number of elements processed by each CTA.

// .x/.y = a0/a1: interval of A input.
// .z/.w = b0/b1: interval of B input.
__host__ __device__ int4 MultiMergePartitionRange(int blockSize, int coop,
	int count, int block) {

	// Round the block index down to a multiple of coop.
	int block2 = ~(coop - 1) & block;
	int intervalSize = blockSize * (coop>> 1);

	int a0 = thrust::min(blockSize * block2, count);
	int a1 = thrust::min(a0 + intervalSize, count);
	int b0 = a1;
	int b1 = thrust::min(b0 + intervalSize, count);
	return make_int4(a0, a1, b0, b1);
}

template<bool UpperBound, typename InputIt1, typename InputIt2, typename Comp>
__host__ __device__ int BlockedMergePath(InputIt1 a_global, int aCount,
	InputIt2 b_global, int bCount, int block, int blockSize, int coop,
	Comp comp) {

	int mp;
	int diag = thrust::min(aCount + bCount, blockSize * block);
	if(coop) {
		int4 ranges = MultiMergePartitionRange(blockSize, coop, aCount, block);
		mp = MergePath<UpperBound>(a_global + ranges.x, ranges.y - ranges.x,
			a_global + ranges.z, ranges.w - ranges.z, diag - ranges.x, comp);
	} else {
		mp = MergePath<UpperBound>(a_global, aCount, b_global, bCount, diag,
			comp);
	}
	return mp;
}


template<bool Duplicates, typename IntT, typename InputIt1, typename InputIt2, 
	typename Comp>
__host__ __device__ int2 BalancedPath(InputIt1 a, int aCount, InputIt2 b,
	int bCount, int diag, IntT levels, Comp comp) {

	typedef typename thrust::iterator_traits<InputIt1>::value_type T;

	int p = MergePath<false>(a, aCount, b, bCount, diag, comp);
	int aIndex = p;
	int bIndex = diag - p;

	bool star = false;
	if(bIndex < bCount) {
		if(Duplicates) {
			T x = b[bIndex];

			// Search for the beginning of the duplicate run in both A and B.
			// Because 
			int aStart = BiasedBinarySearch<false, IntT>(a, aIndex, x, levels, 
				comp);
			int bStart = BiasedBinarySearch<false, IntT>(b, bIndex, x, levels, 
				comp);

			// The distance between the merge path and the lower_bound is the 
			// 'run'. We add up the a- and b- runs and evenly distribute them to
			// get a stairstep path.
			int aRun = aIndex - aStart;
			int bRun = bIndex - bStart;
			int xCount = aRun + bRun;

			// Attempt to advance b and regress a.
			int bAdvance = thrust::max(xCount>> 1, xCount - aRun);
			int bEnd     = thrust::min(bCount, bStart + bAdvance + 1);
			int bRunEnd = BinarySearch<true>(b + bIndex, bEnd - bIndex, x, 
				comp) + bIndex;
			bRun = bRunEnd - bStart;

			bAdvance = thrust::min(bAdvance, bRun);
			int aAdvance = xCount - bAdvance;

			bool roundUp = (aAdvance == bAdvance + 1) && (bAdvance < bRun);
			aIndex = aStart + aAdvance;

			if(roundUp) star = true;
		} else {
			if(aIndex && aCount) {
				T aKey = a[aIndex - 1];
				T bKey = b[bIndex];

				// If the last consumed element in A (aIndex - 1) is the same as
				// the next element in B (bIndex), we're sitting at a starred
				// partition.
				if(!comp(aKey, bKey)) star = true;
			}
		}
	}
	return make_int2(aIndex, star);
}


template<typename IntT, typename InputIt1, typename InputIt2, typename Comp>
__host__ __device__
int2 BalancedPath_jph(InputIt1 a, int aCount,
                      InputIt2 b, int bCount,
                      int diag,
                      IntT levels,
                      Comp comp)
{
  typedef typename thrust::iterator_traits<InputIt1>::value_type T;

  int aIndex = MergePath<false>(a, aCount, b, bCount, diag, comp);
  int bIndex = diag - aIndex;
  
  bool star = false;
  if(bIndex < bCount)
  {
    T x = b[bIndex];
    
    // Search for the beginning of the duplicate run in both A and B.
    int aStart = BiasedBinarySearch<false, IntT>(a, aIndex, x, levels, comp);
    int bStart = BiasedBinarySearch<false, IntT>(b, bIndex, x, levels, comp);
    
    // The distance between x's merge path and its lower_bound is its rank.
    // We add up the a and b ranks and evenly distribute them to
    // get a stairstep path.
    int aRun = aIndex - aStart;
    int bRun = bIndex - bStart;
    int xCount = aRun + bRun;
    
    // Attempt to advance b and regress a.
    int bAdvance = thrust::max(xCount>> 1, xCount - aRun);
    int bEnd     = thrust::min(bCount, bStart + bAdvance + 1);
    int bRunEnd = BinarySearch<true>(b + bIndex, bEnd - bIndex, x, comp) + bIndex;
    bRun = bRunEnd - bStart;
    
    bAdvance = thrust::min(bAdvance, bRun);
    int aAdvance = xCount - bAdvance;
    
    bool roundUp = (aAdvance == bAdvance + 1) && (bAdvance < bRun);
    aIndex = aStart + aAdvance;
    
    if(roundUp) star = true;
  }

  return make_int2(aIndex, star);
}


} // namespace detail


template<typename IntT, typename InputIt1, typename InputIt2, typename Comp>
__host__ __device__
thrust::pair<int,int>
  balanced_path(InputIt1 a, int aCount,
                InputIt2 b, int bCount,
                int diag,
                IntT levels,
                Comp comp)
{
  typedef typename thrust::iterator_traits<InputIt1>::value_type T;

  int aIndex = detail::MergePath<false>(a, aCount, b, bCount, diag, comp);
  int bIndex = diag - aIndex;
  
  bool star = false;
  if(bIndex < bCount)
  {
    T x = b[bIndex];
    
    // Search for the beginning of the duplicate run in both A and B.
    int aStart = detail::BiasedBinarySearch<false, IntT>(a, aIndex, x, levels, comp);
    int bStart = detail::BiasedBinarySearch<false, IntT>(b, bIndex, x, levels, comp);
    
    // The distance between x's merge path and its lower_bound is its rank.
    // We add up the a and b ranks and evenly distribute them to
    // get a stairstep path.
    int aRun = aIndex - aStart;
    int bRun = bIndex - bStart;
    int xCount = aRun + bRun;
    
    // Attempt to advance b and regress a.
    int bAdvance = thrust::max(xCount>> 1, xCount - aRun);
    int bEnd     = thrust::min(bCount, bStart + bAdvance + 1);
    int bRunEnd = detail::BinarySearch<true>(b + bIndex, bEnd - bIndex, x, comp) + bIndex;
    bRun = bRunEnd - bStart;
    
    bAdvance = thrust::min(bAdvance, bRun);
    int aAdvance = xCount - bAdvance;
    
    bool roundUp = (aAdvance == bAdvance + 1) && (bAdvance < bRun);
    aIndex = aStart + aAdvance;
    
    if(roundUp) star = true;
  }

  return thrust::make_pair(aIndex, (diag - aIndex) + star);
}

