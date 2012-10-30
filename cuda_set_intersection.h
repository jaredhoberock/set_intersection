#pragma once
#include "balanced_path.h"
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/pair.h>
#include <thrust/system/cuda/detail/detail/uninitialized.h>
#include <thrust/system/cuda/error.h>
#include <thrust/detail/util/blocking.h>
#include <vector>

namespace set_intersection_detail
{


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


// XXX optimize me!
template<typename Iterator>
inline __device__
  void blockwise_inplace_exclusive_scan(Iterator first, Iterator last)
{
  if(threadIdx.x == 0)
  {
    typename thrust::iterator_value<Iterator>::type sum = 0;

    for(; first != last; ++first)
    {
      typename thrust::iterator_value<Iterator>::type temp = *first;
      *first = sum;
      sum += temp;
    }
  }

  __syncthreads();
}


// XXX optimize me!
template<typename Iterator>
inline __device__
  typename thrust::iterator_value<Iterator>::type
    blockwise_reduce(Iterator first, Iterator last)
{
  typename thrust::iterator_value<Iterator>::type result = 0;
  for(; first != last; ++first)
  {
    result += *first;
  }

  __syncthreads();

  return result;
}


template<int block_size, int work_per_thread, typename InputIterator1, typename InputIterator2, typename Compare>
inline __device__
  unsigned int blockwise_count_set_intersection(InputIterator1 first1, InputIterator1 last1,
                                                InputIterator2 first2, InputIterator2 last2,
                                                Compare comp)
{
  int thread_idx = threadIdx.x;
  int n1 = last1 - first1;
  int n2 = last2 - first2;

  __shared__ thrust::system::cuda::detail::detail::uninitialized_array<thrust::pair<int,int>, block_size + 1> s_input_partition_offsets;

  if(thread_idx == 0)
  {
    s_input_partition_offsets[block_size] = thrust::make_pair(n1,n2);
  }

  __syncthreads();

  // find partition offsets
  int diag = min(n1 + n2, thread_idx * work_per_thread);
  s_input_partition_offsets[thread_idx] = balanced_path(first1, n1, first2, n2, diag, 2, comp);

  __syncthreads();

  __shared__ unsigned int s_thread_output_size[block_size];

  // serially count a set intersection
  thrust::pair<int,int> thread_input_begin = s_input_partition_offsets[thread_idx];
  thrust::pair<int,int> thread_input_end   = s_input_partition_offsets[thread_idx + 1];

  s_thread_output_size[thread_idx] =
    serial_bounded_count_set_intersection(work_per_thread,
                                          first1 + thread_input_begin.first,  first1 + thread_input_end.first,
                                          first2 + thread_input_begin.second, first2 + thread_input_end.second,
                                          comp);

  __syncthreads();

  // reduce per-thread counts
  return blockwise_reduce(&s_thread_output_size[0], &s_thread_output_size[block_size]);
}


template<int block_size, int work_per_thread, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Compare>
inline __device__
  OutputIterator blockwise_set_intersection(InputIterator1 first1, InputIterator1 last1,
                                            InputIterator2 first2, InputIterator2 last2,
                                            OutputIterator result,
                                            Compare comp)
{
  int thread_idx = threadIdx.x;
  int n1 = last1 - first1;
  int n2 = last2 - first2;

  __shared__ thrust::system::cuda::detail::detail::uninitialized_array<thrust::pair<int,int>, block_size + 1> s_input_partition_offsets;

  if(thread_idx == 0)
  {
    s_input_partition_offsets.back() = thrust::make_pair(n1,n2);
  }

  __syncthreads();

  // find partition offsets
  int diag = thrust::min(n1 + n2, thread_idx * work_per_thread);
  s_input_partition_offsets[thread_idx] = balanced_path(first1, n1, first2, n2, diag, 2, comp);

  __syncthreads();

  typedef typename thrust::iterator_value<InputIterator1>::type value_type;

  __shared__ thrust::system::cuda::detail::detail::uninitialized_array<unsigned int, block_size + 1> s_thread_output_size;

  thrust::pair<int,int> thread_input_begin = s_input_partition_offsets[thread_idx];
  thrust::pair<int,int> thread_input_end   = s_input_partition_offsets[thread_idx + 1];

  value_type   sparse_results[work_per_thread];
  unsigned int active_mask =
    serial_bounded_sparse_set_intersection(work_per_thread,
                                           first1 + thread_input_begin.first,  first1 + thread_input_end.first,
                                           first2 + thread_input_begin.second, first2 + thread_input_end.second,
                                           sparse_results,
                                           comp);

  s_thread_output_size[thread_idx] = __popc(active_mask);

  __syncthreads();

  // scan to turn per-thread counts into output indices
  // XXX the size of this scan is block_size + 1
  blockwise_inplace_exclusive_scan(s_thread_output_size.begin(), s_thread_output_size.end());

  serial_bounded_copy_if(work_per_thread, sparse_results, active_mask, result + s_thread_output_size[thread_idx]);

  __syncthreads();

  return result + s_thread_output_size[block_size];
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

  // XXX gmem -> smem here

  unsigned int count =
    blockwise_count_set_intersection<threads_per_block,work_per_thread>(first1 + block_input_begin.first,  first1 + block_input_end.first,
                                                                        first2 + block_input_begin.second, first2 + block_input_end.second,
                                                                        comp);

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

  // XXX consider streaming partitions which are larger than block_size * work_per_thread
  //     to do this, each block break its partition into subpartitions of size <= threads_per_block * work_per_thread
  //     to find the end of each subpartition, it would do a balanced_path search
  // XXX is there a way to avoid this search? each block proceeds sequentially

  // XXX gmem -> smem here
  
  blockwise_set_intersection<threads_per_block,work_per_thread>(first1 + block_input_begin.first,  first1 + block_input_end.first,
                                                                first2 + block_input_begin.second, first2 + block_input_end.second,
                                                                result + output_partition_offsets[block_idx],
                                                                comp);

  // XXX smem -> gmem here
}


} // end set_intersection_detail


template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Compare>
  OutputIterator my_set_intersection(InputIterator1 first1, InputIterator1 last1,
                                     InputIterator2 first2, InputIterator2 last2,
                                     OutputIterator result,
                                     Compare comp)
{
  const int n1 = last1 - first1;
  const int n2 = last2 - first2;

  const int threads_per_block = 128;
  const int work_per_thread = 15;
  const int work_per_block = threads_per_block * work_per_thread;
  const int num_partitions = thrust::detail::util::divide_ri(n1 + n2, work_per_block);

  // find input partition offsets
  // +1 to handle the end of the input elegantly
  thrust::device_vector<thrust::pair<int,int> > input_partition_offsets(num_partitions + 1);
  set_intersection_detail::find_partition_offsets<int>(input_partition_offsets.size(), work_per_block, first1, last1, first2, last2, input_partition_offsets.begin(), comp);

  // find output partition offsets
  // +1 to store the total size of the total
  thrust::device_vector<int> output_partition_offsets(num_partitions + 1);
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

  return result + output_partition_offsets.back();
}

