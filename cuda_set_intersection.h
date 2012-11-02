#pragma once
#include "balanced_path.h"
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/pair.h>
#include <thrust/system/cuda/detail/detail/uninitialized.h>
#include <thrust/system/cuda/error.h>
#include <thrust/detail/util/blocking.h>
#include <thrust/detail/temporary_array.h>
#include <vector>
#include <cassert>

namespace set_intersection_detail
{

using thrust::system::cuda::detail::detail::uninitialized_array;


template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Compare>
inline __device__
  unsigned int serial_bounded_sparse_set_intersection(int max_input_size,
                                                      InputIterator1 first1, InputIterator1 last1,
                                                      InputIterator2 first2, InputIterator2 last2,
                                                      OutputIterator result,
                                                      Compare comp)
{
  unsigned int active_mask = 0;
  unsigned int active_bit = 1;

  while(first1 != last1 && first2 != last2)
  {
    if(comp(*first1,*first2))
    {
      ++first1;
    } // end if
    else if(comp(*first2,*first1))
    {
      ++first2;
    } // end else if
    else
    {
      *result = *first1;
      ++first1;
      ++first2;
      active_mask |= active_bit;
    } // end else

    ++result;
    active_bit <<= 1;
  } // end while

  return active_mask;
}


template<typename InputIterator1, typename InputIterator2, typename Compare>
inline __device__
  unsigned int serial_bounded_count_set_intersection(int max_input_size,
                                                     InputIterator1 first1, InputIterator1 last1,
                                                     InputIterator2 first2, InputIterator2 last2,
                                                     Compare comp)
{
  unsigned int result = 0;

  while(first1 != last1 && first2 != last2)
  {
    if(comp(*first1,*first2))
    {
      ++first1;
    } // end if
    else if(comp(*first2,*first1))
    {
      ++first2;
    } // end else if
    else
    {
      ++result;
      ++first1;
      ++first2;
    } // end else
  } // end while

  return result;
}


template<typename InputIterator, typename OutputIterator>
inline __device__
  OutputIterator serial_bounded_copy_if(unsigned int max_input_size,
                                        InputIterator first,
                                        unsigned int mask,
                                        OutputIterator result)
{
  for(unsigned int i = 0; i < max_input_size; ++i, ++first)
  {
    if((1<<i) & mask)
    {
      *result = *first;
      ++result;
    }
  }

  return result;
}


template<typename Size, typename InputIterator1, typename InputIterator2, typename Compare>
  struct find_partition_offsets_functor
{
  Size partition_size;
  InputIterator1 first1;
  InputIterator2 first2;
  Size n1, n2;
  Compare comp;

  find_partition_offsets_functor(Size partition_size,
                                 InputIterator1 first1, InputIterator1 last1,
                                 InputIterator2 first2, InputIterator2 last2,
                                 Compare comp)
    : partition_size(partition_size),
      first1(first1), first2(first2),
      n1(last1 - first1), n2(last2 - first2),
      comp(comp)
  {}

  inline __host__ __device__
  thrust::pair<Size,Size> operator()(Size i) const
  {
    Size diag = min(n1 + n2, i * partition_size);

    // XXX the correctness of balanced_path depends critically on the ll suffix below
    //     why???
    return balanced_path(first1, n1, first2, n2, diag, 4ll, comp);
  }
};


template<typename Size, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Compare>
  OutputIterator find_partition_offsets(Size num_partitions,
                                        Size partition_size,
                                        InputIterator1 first1, InputIterator1 last1,
                                        InputIterator2 first2, InputIterator2 last2,
                                        OutputIterator result,
                                        Compare comp)
{
  find_partition_offsets_functor<Size,InputIterator1,InputIterator2,Compare> f(partition_size, first1, last1, first2, last2, comp);

  return thrust::transform(thrust::counting_iterator<Size>(0),
                           thrust::counting_iterator<Size>(num_partitions),
                           result,
                           f);
}


template<typename RandomAccessIterator1, typename Size, typename RandomAccessIterator2>
inline __device__
RandomAccessIterator2 blockwise_copy_n(RandomAccessIterator1 first, Size n, RandomAccessIterator2 result)
{
  for(unsigned int i = threadIdx.x; i < n; i += blockDim.x)
  {
    result[i] = first[i];
  }

  __syncthreads();

  return result + n;
}


template<typename RandomAccessIterator, typename BinaryFunction>
inline __device__
void blockwise_inplace_inclusive_scan(RandomAccessIterator first, BinaryFunction op)
{
  typename thrust::iterator_value<RandomAccessIterator>::type x = first[threadIdx.x];

  for(unsigned int offset = 1; offset < blockDim.x; offset *= 2)
  {
    if(threadIdx.x >= offset)
    {
      x = op(first[threadIdx.x - offset], x);
    }

    __syncthreads();

    first[threadIdx.x] = x;

    __syncthreads();
  }
}


template<typename RandomAccessIterator>
inline __device__
void blockwise_inplace_inclusive_scan(RandomAccessIterator first)
{
  blockwise_inplace_inclusive_scan(first, thrust::plus<typename thrust::iterator_value<RandomAccessIterator>::type>());
}


// XXX improve this
template<typename RandomAccessIterator, typename T, typename BinaryFunction>
inline __device__
typename thrust::iterator_value<RandomAccessIterator>::type
  blockwise_inplace_exclusive_scan(RandomAccessIterator first, T init, BinaryFunction op)
{
  // perform an inclusive scan, then shift right
  blockwise_inplace_inclusive_scan(first, op);

  typename thrust::iterator_value<RandomAccessIterator>::type carry = first[blockDim.x - 1];

  __syncthreads();

  typename thrust::iterator_value<RandomAccessIterator>::type left = (threadIdx.x == 0) ? init : first[threadIdx.x - 1];

  __syncthreads();

  first[threadIdx.x] = left;

  __syncthreads();

  return carry;
}



template<typename Iterator, typename T>
inline __device__
  typename thrust::iterator_value<Iterator>::type
    blockwise_inplace_exclusive_scan(Iterator first, T init)
{
  return blockwise_inplace_exclusive_scan(first, init, thrust::plus<typename thrust::iterator_value<Iterator>::type>());
}


template<int block_size, typename T>
inline __device__
T blockwise_right_neighbor(const T &x, const T &boundary)
{
  // stage this shift to conserve smem
  const unsigned int storage_size = block_size / 2;
  __shared__ uninitialized_array<T,storage_size> shared;

  T result = x;

  unsigned int tid = threadIdx.x;

  if(0 < tid && tid <= storage_size)
  {
    shared[tid - 1] = x;
  }

  __syncthreads();

  if(tid < storage_size)
  {
    result = shared[tid];
  }

  __syncthreads();
  
  tid -= storage_size;
  if(0 < tid && tid <= storage_size)
  {
    shared[tid - 1] = x;
  }
  else if(tid == 0)
  {
    shared[storage_size-1] = boundary;
  }

  __syncthreads();

  if(tid < storage_size)
  {
    result = shared[tid];
  }

  __syncthreads();

  return result;
}


template<int block_size, int work_per_thread, typename InputIterator1, typename InputIterator2, typename Compare>
inline __device__
  unsigned int blockwise_bounded_count_set_intersection_n(InputIterator1 first1, int n1,
                                                          InputIterator2 first2, int n2,
                                                          Compare comp)
{
  int thread_idx = threadIdx.x;

  // find partition offsets
  int diag = min(n1 + n2, thread_idx * work_per_thread);
  thrust::pair<short,short> thread_input_begin = balanced_path(first1, n1, first2, n2, diag, 2, comp);
  thrust::pair<short,short> thread_input_end   = blockwise_right_neighbor<block_size>(thread_input_begin, thrust::make_pair<short,short>(n1,n2));

  __shared__ thrust::detail::uint16_t s_thread_output_size[block_size];

  // work_per_thread + 1 to accomodate a "starred" partition returned from balanced_path above
  s_thread_output_size[thread_idx] =
    serial_bounded_count_set_intersection(work_per_thread + 1,
                                          first1 + thread_input_begin.first,  first1 + thread_input_end.first,
                                          first2 + thread_input_begin.second, first2 + thread_input_end.second,
                                          comp);

  __syncthreads();

  // reduce per-thread counts
  blockwise_inplace_inclusive_scan(s_thread_output_size);
  return s_thread_output_size[blockDim.x - 1];
}


template<int block_size, int work_per_thread, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Compare>
inline __device__
  OutputIterator blockwise_bounded_set_intersection_n(InputIterator1 first1, int n1,
                                                      InputIterator2 first2, int n2,
                                                      OutputIterator result,
                                                      Compare comp)
{
  int thread_idx = threadIdx.x;
  
  // find partition offsets
  int diag = thrust::min(n1 + n2, thread_idx * work_per_thread);
  thrust::pair<short,short> thread_input_begin = balanced_path(first1, n1, first2, n2, diag, 2, comp);
  thrust::pair<short,short> thread_input_end   = blockwise_right_neighbor<block_size>(thread_input_begin, thrust::make_pair<short,short>(n1,n2));

  typedef typename thrust::iterator_value<InputIterator1>::type value_type;
  // +1 to accomodate a "starred" partition returned from balanced_path above
  uninitialized_array<value_type, work_per_thread + 1> sparse_result;
  unsigned int active_mask =
    serial_bounded_sparse_set_intersection(work_per_thread + 1,
                                           first1 + thread_input_begin.first,  first1 + thread_input_end.first,
                                           first2 + thread_input_begin.second, first2 + thread_input_end.second,
                                           sparse_result.begin(),
                                           comp);

  __shared__ thrust::detail::uint16_t s_thread_output_size[block_size];
  s_thread_output_size[thread_idx] = __popc(active_mask);

  __syncthreads();

  // scan to turn per-thread counts into output indices
  thrust::detail::uint16_t block_output_size = blockwise_inplace_exclusive_scan(s_thread_output_size, 0u);

  serial_bounded_copy_if(work_per_thread + 1, sparse_result.begin(), active_mask, result + s_thread_output_size[thread_idx]);

  __syncthreads();

  return result + block_output_size;
}


template<int threads_per_block, int work_per_thread, typename InputIterator1, typename InputIterator2, typename InputIterator3, typename OutputIterator, typename Compare>
  __global__ void my_count_set_intersection_kernel(InputIterator1 input_partition_offsets,
                                                   InputIterator2 first1,
                                                   InputIterator3 first2,
                                                   OutputIterator result,
                                                   Compare comp)
{
  int block_idx = blockIdx.x;

  // each block counts a set op across a partition
  thrust::pair<int,int> block_input_begin = input_partition_offsets[block_idx];
  thrust::pair<int,int> block_input_end   = input_partition_offsets[block_idx + 1];

  thrust::pair<int,int> remaining_input_size = thrust::make_pair(block_input_end.first  - block_input_begin.first,
                                                                 block_input_end.second - block_input_begin.second);

  // advance first1 & first2
  first1 += block_input_begin.first;
  first2 += block_input_begin.second;

  // iterate until the input is consumed
  unsigned int count = 0;
  while(remaining_input_size.first + remaining_input_size.second > 0)
  {
    // find the end of this subpartition's input
    // -1 to accomodate "starred" partitions
    int max_subpartition_size = threads_per_block * work_per_thread - 1;
    int diag = min(remaining_input_size.first + remaining_input_size.second, max_subpartition_size);
    thrust::pair<int,int> subpartition_size = balanced_path(first1, remaining_input_size.first, first2, remaining_input_size.second, diag, 4ll, comp);
  
    // load the input into __shared__ storage
    typedef typename thrust::iterator_value<InputIterator2>::type value_type;
    __shared__ uninitialized_array<value_type, threads_per_block * work_per_thread> s_input;
  
    value_type *s_input_end1 = blockwise_copy_n(first1, subpartition_size.first,  s_input.begin());
    value_type *s_input_end2 = blockwise_copy_n(first2, subpartition_size.second, s_input_end1);
  
    count += blockwise_bounded_count_set_intersection_n<threads_per_block,work_per_thread>(s_input.begin(), subpartition_size.first,
                                                                                           s_input_end1,    subpartition_size.second,
                                                                                           comp);

    // advance input
    first1 += subpartition_size.first;
    first2 += subpartition_size.second;

    // decrement remaining size
    remaining_input_size.first  -= subpartition_size.first;
    remaining_input_size.second -= subpartition_size.second;
  }

  if(threadIdx.x == 0)
  {
    result[block_idx] = count;
  }
}


template<int threads_per_block, int work_per_thread, typename InputIterator1, typename InputIterator2, typename InputIterator3, typename InputIterator4, typename OutputIterator, typename Compare>
__global__
  void my_set_intersection_kernel(InputIterator1 input_partition_offsets,
                                  InputIterator2 first1,
                                  InputIterator3 first2,
                                  InputIterator4 output_partition_offsets,
                                  OutputIterator result,
                                  Compare comp)
{
  int block_idx = blockIdx.x;

  // each block does a set op across a partition
  thrust::pair<int,int> block_input_begin = input_partition_offsets[block_idx];
  thrust::pair<int,int> block_input_end   = input_partition_offsets[block_idx + 1];

  thrust::pair<int,int> remaining_input_size = thrust::make_pair(block_input_end.first  - block_input_begin.first,
                                                                 block_input_end.second - block_input_begin.second);

  // advance iterators
  first1 += block_input_begin.first;
  first2 += block_input_begin.second;
  result += output_partition_offsets[block_idx];

  // iterate until the input is consumed
  while(remaining_input_size.first + remaining_input_size.second > 0)
  {
    // find the end of this subpartition's input
    // -1 to accomodate "starred" partitions
    int max_subpartition_size = threads_per_block * work_per_thread - 1;
    int diag = min(remaining_input_size.first + remaining_input_size.second, max_subpartition_size);
    thrust::pair<int,int> subpartition_size = balanced_path(first1, remaining_input_size.first, first2, remaining_input_size.second, diag, 4ll, comp);
    
    // load the input into __shared__ storage
    typedef typename thrust::iterator_value<InputIterator2>::type value_type;
    __shared__ uninitialized_array<value_type, threads_per_block * work_per_thread> s_input;

    value_type *s_input_end1 = blockwise_copy_n(first1, subpartition_size.first,  s_input.begin());
    value_type *s_input_end2 = blockwise_copy_n(first2, subpartition_size.second, s_input_end1);

    result = blockwise_bounded_set_intersection_n<threads_per_block,work_per_thread>(s_input.begin(), subpartition_size.first,
                                                                                     s_input_end1,    subpartition_size.second,
                                                                                     result,
                                                                                     comp);

    // advance input
    first1 += subpartition_size.first;
    first2 += subpartition_size.second;

    // decrement remaining size
    remaining_input_size.first  -= subpartition_size.first;
    remaining_input_size.second -= subpartition_size.second;
  }
}


} // end set_intersection_detail


template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Compare>
  OutputIterator my_set_intersection(InputIterator1 first1, InputIterator1 last1,
                                     InputIterator2 first2, InputIterator2 last2,
                                     OutputIterator result,
                                     Compare comp)
{
  typedef thrust::cuda::tag System;
  System system;
  using thrust::system::cuda::detail::device_properties;

  const int n1 = last1 - first1;
  const int n2 = last2 - first2;

  const int work_per_thread         = 15;
  const int threads_per_block       = 128;
  const int work_per_block = threads_per_block * work_per_thread;

  // -1 because balanced_path adds a single element to the end of a "starred" partition, increasing its size by one
  const int maximum_partition_size = work_per_block - 1;
  //const int max_num_blocks = device_properties().maxGridSize[0];
  //const int num_partitions = min(max_num_blocks, thrust::detail::util::divide_ri(n1 + n2, maximum_partition_size));
  const int max_num_blocks = device_properties().maxGridSize[0];
  const int num_partitions = thrust::detail::util::divide_ri(n1 + n2, maximum_partition_size);

  // find input partition offsets
  // +1 to handle the end of the input elegantly
  thrust::detail::temporary_array<thrust::pair<int,int>, System> input_partition_offsets(0, system, num_partitions + 1);
  set_intersection_detail::find_partition_offsets<int>(input_partition_offsets.size(), maximum_partition_size, first1, last1, first2, last2, input_partition_offsets.begin(), comp);

  // find output partition offsets
  // +1 to store the total size of the total
  thrust::detail::temporary_array<int, System> output_partition_offsets(0, system, num_partitions + 1);
  set_intersection_detail::my_count_set_intersection_kernel<threads_per_block,work_per_thread><<<num_partitions,threads_per_block>>>(input_partition_offsets.begin(), first1, first2, output_partition_offsets.begin(), comp);
  cudaError_t error = cudaGetLastError();
  if(error)
  {
    throw thrust::system_error(error, thrust::cuda_category(), std::string("CUDA error after my_count_set_intersection_kernel"));
  }

  // turn the counts into offsets
  thrust::exclusive_scan(output_partition_offsets.begin(), output_partition_offsets.end(), output_partition_offsets.begin(), 0);

  // run the set op kernel
  set_intersection_detail::my_set_intersection_kernel<threads_per_block,work_per_thread><<<num_partitions,threads_per_block>>>(input_partition_offsets.begin(), first1, first2, output_partition_offsets.begin(), result, comp);
  error = cudaThreadSynchronize();
  if(error)
  {
    throw thrust::system_error(error, thrust::cuda_category(), std::string("CUDA error after my_set_intersection_kernel"));
  }

  return result + output_partition_offsets[num_partitions];
}

