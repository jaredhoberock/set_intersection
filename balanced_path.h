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
__host__ __device__ int MergePath(It1 a, int aCount, It2 b, int bCount, int diag, Comp comp)
{
  typedef typename thrust::iterator_traits<It1>::value_type T;
  
  int begin = thrust::max(0, diag - bCount);
  int end   = thrust::min(diag, aCount);
  
  while(begin < end) 
  {
    int mid = (begin + end)>> 1;
    T aKey = a[mid];
    T bKey = b[diag - 1 - mid];
    bool pred = UpperBound ? comp(aKey, bKey) : !comp(bKey, aKey);
    if(pred) begin = mid + 1;
    else end = mid;
  }
  return begin;
}


} // namespace detail


template<typename RandomAccessIterator1, typename Size1, typename RandomAccessIterator2, typename Size2, typename Compare>
__host__ __device__
thrust::pair<Size1,Size1>
  balanced_path(RandomAccessIterator1 first1, Size1 n1,
                RandomAccessIterator2 first2, Size1 n2,
                Size1 diag,
                Size2 levels,
                Compare comp)
{
  typedef typename thrust::iterator_traits<RandomAccessIterator1>::value_type T;

  Size1 aIndex = detail::MergePath<false>(first1, n1, first2, n2, diag, comp);
  Size1 bIndex = diag - aIndex;
  
  bool star = false;
  if(bIndex < n2)
  {
    T x = first2[bIndex];
    
    // Search for the beginning of the duplicate run in both A and B.
    Size1 aStart = detail::BiasedBinarySearch<false>(first1, aIndex, x, levels, comp);
    Size1 bStart = detail::BiasedBinarySearch<false>(first2, bIndex, x, levels, comp);
    
    // The distance between x's merge path and its lower_bound is its rank.
    // We add up the a and b ranks and evenly distribute them to
    // get a stairstep path.
    Size1 aRun = aIndex - aStart;
    Size1 bRun = bIndex - bStart;
    Size1 xCount = aRun + bRun;
    
    // Attempt to advance b and regress a.
    Size1 bAdvance = thrust::max(xCount >> 1, xCount - aRun);
    Size1 bEnd     = thrust::min<Size1>(n2, bStart + bAdvance + 1);
    Size1 bRunEnd = detail::BinarySearch<true>(first2 + bIndex, bEnd - bIndex, x, comp) + bIndex;
    bRun = bRunEnd - bStart;
    
    bAdvance = thrust::min(bAdvance, bRun);
    Size1 aAdvance = xCount - bAdvance;
    
    bool roundUp = (aAdvance == bAdvance + 1) && (bAdvance < bRun);
    aIndex = aStart + aAdvance;
    
    if(roundUp) star = true;
  }

  return thrust::make_pair(aIndex, (diag - aIndex) + star);
}

