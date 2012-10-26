#include "utility.h"
#include "balanced_path.h"
#include <vector>
#include <algorithm>
#include <utility>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>


template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Compare>
  OutputIterator serial_set_intersection(InputIterator1 first1, InputIterator1 last1,
                                         InputIterator2 first2, InputIterator2 last2,
                                         OutputIterator result,
                                         Compare comp)
{
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
      ++result;
    } // end else
  } // end while

  return result;
}


template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Compare>
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
  OutputIterator serial_bounded_copy_if(unsigned int max_input_size,
                                        InputIterator first,
                                        unsigned int mask,
                                        OutputIterator result)
{
  for(int i = 0; i < max_input_size; ++i, ++first)
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

  return thrust::transform(thrust::counting_iterator<Size,thrust::host_system_tag>(0),
                           thrust::counting_iterator<Size,thrust::host_system_tag>(num_partitions),
                           result,
                           f);
}

template<typename L, typename R>
  inline __host__ __device__ L divide_ri(const L x, const R y)
{
  return (x + (y - 1)) / y;
}


inline __host__ __device__
unsigned int pop_count(unsigned int x)
{
  unsigned int result; // c accumulates the total bits set in v

  for(result = 0; x; x >>= 1)
  {
    result += x & 1;
  }

  return result;
}


template<typename T>
  class per_thread
{
  public:
    per_thread(size_t block_size)
      : m_impl(block_size)
    {}

    template<typename Arg>
    per_thread(size_t block_size, const Arg &arg)
      : m_impl(block_size)
    {
      for(unsigned int i = 0;
          i < block_size;
          ++i)
      {
        m_impl[i] = T(arg);
      }
    }

    T &operator[](unsigned int thread_idx)
    {
      return m_impl[thread_idx];
    }

    typename std::vector<T>::iterator begin()
    {
      return m_impl.begin();
    }

    typename std::vector<T>::iterator end()
    {
      return m_impl.end();
    }

  private:
    std::vector<T> m_impl;
};


template<typename Iterator>
  void blockwise_inplace_exclusive_scan(Iterator first, Iterator last)
{
  thrust::exclusive_scan(first, last, first);
}

template<typename Iterator>
  typename thrust::iterator_value<Iterator>::type blockwise_reduce(Iterator first, Iterator last)
{
  return thrust::reduce(first, last);
}


inline void sync_threads()
{
  ;
}


template<typename InputIterator1, typename InputIterator2, typename Compare>
  unsigned int blockwise_count_set_intersection(int block_size,
                                                int work_per_thread,
                                                InputIterator1 first1, InputIterator1 last1,
                                                InputIterator2 first2, InputIterator2 last2,
                                                Compare comp)
{
  int n1 = last1 - first1;
  int n2 = last2 - first2;

  /* __shared__ */ std::vector<thrust::pair<int,int> > s_input_partition_offsets(block_size + 1);

  for(int thread_idx = 0;
      thread_idx < block_size;
      ++thread_idx)
  {
    if(thread_idx == 0)
    {
      s_input_partition_offsets.back() = thrust::make_pair(n1,n2);
    }
  }

  sync_threads();

  // find partition offsets
  for(int thread_idx = 0;
      thread_idx < block_size;
      ++thread_idx)
  {
    int diag = min(n1 + n2, thread_idx * work_per_thread);
    s_input_partition_offsets[thread_idx] = balanced_path(first1, n1, first2, n2, diag, 2, comp);
  }

  sync_threads();

  /* __shared__ */ std::vector<unsigned int> s_thread_output_size(block_size);

  // serially count a set intersection
  for(int thread_idx = 0;
      thread_idx < block_size;
      ++thread_idx)
  {
    thrust::pair<int,int> thread_input_begin = s_input_partition_offsets[thread_idx];
    thrust::pair<int,int> thread_input_end   = s_input_partition_offsets[thread_idx + 1];

    s_thread_output_size[thread_idx] =
      serial_bounded_count_set_intersection(work_per_thread,
                                            first1 + thread_input_begin.first,  first1 + thread_input_end.first,
                                            first2 + thread_input_begin.second, first2 + thread_input_end.second,
                                            comp);
  }

  sync_threads();

  // reduce per-thread counts
  return blockwise_reduce(s_thread_output_size.begin(), s_thread_output_size.end());
}


template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Compare>
  OutputIterator blockwise_set_intersection(int block_size,
                                            int work_per_thread,
                                            InputIterator1 first1, InputIterator1 last1,
                                            InputIterator2 first2, InputIterator2 last2,
                                            OutputIterator result,
                                            Compare comp)
{
  int n1 = last1 - first1;
  int n2 = last2 - first2;

  /* __shared__ */ std::vector<thrust::pair<int,int> > s_input_partition_offsets(block_size + 1);

  for(int thread_idx = 0;
      thread_idx < block_size;
      ++thread_idx)
  {
    if(thread_idx == 0)
    {
      s_input_partition_offsets.back() = thrust::make_pair(n1,n2);
    }
  }

  sync_threads();

  // find partition offsets
  for(int thread_idx = 0;
      thread_idx < block_size;
      ++thread_idx)
  {
    int diag = min(n1 + n2, thread_idx * work_per_thread);
    s_input_partition_offsets[thread_idx] = balanced_path(first1, n1, first2, n2, diag, 2, comp);
  }

  sync_threads();

  typedef typename thrust::iterator_value<InputIterator1>::type value_type;

  per_thread<std::vector<value_type> > sparse_results(block_size, work_per_thread);
  per_thread<unsigned int>             active_mask(block_size);

  /* __shared__ */ std::vector<unsigned int> s_thread_output_size(block_size + 1);

  for(int thread_idx = 0;
      thread_idx < block_size;
      ++thread_idx)
  {
    thrust::pair<int,int> thread_input_begin = s_input_partition_offsets[thread_idx];
    thrust::pair<int,int> thread_input_end   = s_input_partition_offsets[thread_idx + 1];

    active_mask[thread_idx] =
      serial_bounded_sparse_set_intersection(work_per_thread,
                                             first1 + thread_input_begin.first,  first1 + thread_input_end.first,
                                             first2 + thread_input_begin.second, first2 + thread_input_end.second,
                                             sparse_results[thread_idx].begin(),
                                             comp);

    s_thread_output_size[thread_idx] = pop_count(active_mask[thread_idx]);
  }

  sync_threads();

  // scan to turn per-thread counts into output indices
  // XXX the size of this scan is block_size + 1
  blockwise_inplace_exclusive_scan(s_thread_output_size.begin(), s_thread_output_size.end());

  for(int thread_idx = 0;
      thread_idx < block_size;
      ++thread_idx)
  {
    serial_bounded_copy_if(work_per_thread, sparse_results[thread_idx].begin(), active_mask[thread_idx], result + s_thread_output_size[thread_idx]);
  }

  sync_threads();

  return result + s_thread_output_size[block_size];
}


template<typename Size, typename InputIterator1, typename InputIterator2, typename InputIterator3, typename OutputIterator, typename Compare>
  void my_count_set_intersection_kernel(Size num_blocks,
                                        Size threads_per_block,
                                        Size work_per_thread,
                                        InputIterator1 input_partition_offsets,
                                        InputIterator2 first1,
                                        InputIterator3 first2,
                                        OutputIterator result,
                                        Compare comp)
{
  // each block counts a set op across a partition
  for(int block_idx = 0;
      block_idx < num_blocks;
      ++block_idx)
  {
    thrust::pair<int,int> block_input_begin = input_partition_offsets[block_idx];
    thrust::pair<int,int> block_input_end   = input_partition_offsets[block_idx + 1];

    // XXX gmem -> smem here

    result[block_idx] = blockwise_count_set_intersection(threads_per_block,
                                                         work_per_thread,
                                                         first1 + block_input_begin.first,  first1 + block_input_end.first,
                                                         first2 + block_input_begin.second, first2 + block_input_end.second,
                                                         comp);
  }
}


template<typename Size, typename InputIterator1, typename InputIterator2, typename InputIterator3, typename InputIterator4, typename OutputIterator, typename Compare>
  void my_set_intersection_kernel(Size num_blocks,
                                  Size threads_per_block,
                                  Size work_per_thread,
                                  InputIterator1 input_partition_offsets,
                                  InputIterator2 first1,
                                  InputIterator3 first2,
                                  InputIterator4 output_partition_offsets,
                                  OutputIterator result,
                                  Compare comp)
{
  // each block does a set op across a partition
  for(int block_idx = 0;
      block_idx < num_blocks;
      ++block_idx)
  {
    thrust::pair<int,int> block_input_begin = input_partition_offsets[block_idx];
    thrust::pair<int,int> block_input_end   = input_partition_offsets[block_idx + 1];

    // XXX consider streaming partitions which are larger than block_size * work_per_thread
    //     to do this, each block break its partition into subpartitions of size <= threads_per_block * work_per_thread
    //     to find the end of each subpartition, it would do a balanced_path search
    // XXX is there a way to avoid this search? each block proceeds sequentially

    // XXX gmem -> smem here

    blockwise_set_intersection(threads_per_block,
                               work_per_thread,
                               first1 + block_input_begin.first,  first1 + block_input_end.first,
                               first2 + block_input_begin.second, first2 + block_input_end.second,
                               result + output_partition_offsets[block_idx],
                               comp);

    // XXX smem -> gmem here
  }
}


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
  const int num_partitions = divide_ri(n1 + n2, work_per_block);

  // find input partition offsets
  std::vector<thrust::pair<int,int> > input_partition_offsets(num_partitions + 1);
  find_partition_offsets<int>(input_partition_offsets.size(), work_per_block, first1, last1, first2, last2, input_partition_offsets.begin(), comp);

  // find output partition offsets
  std::vector<int> output_partition_offsets(num_partitions + 1);
  my_count_set_intersection_kernel(num_partitions, threads_per_block, work_per_thread, input_partition_offsets.begin(), first1, first2, output_partition_offsets.begin(), comp);

  // turn the counts into offsets
  thrust::exclusive_scan(output_partition_offsets.begin(), output_partition_offsets.end(), output_partition_offsets.begin(), 0);

  // run the set op kernel
  my_set_intersection_kernel(num_partitions, threads_per_block, work_per_thread, input_partition_offsets.begin(), first1, first2, output_partition_offsets.begin(), result, comp);

  return result + output_partition_offsets[num_partitions];
}


template<typename T>
void test(int size)
{
  int a_size = size;
  int b_size = size + 1;
  int input_size = a_size + b_size;

  typedef thrust::less<T> Compare;
  Compare comp;

  thrust::host_vector<T> a, b;
  random_sorted_vector(a_size, comp, a);
  random_sorted_vector(b_size, comp, b);

  thrust::host_vector<T> o(input_size, T(13));

  o.erase(my_set_intersection(a.begin(), a.end(), b.begin(), b.end(), o.begin(), comp), o.end());

  verify(a, b, o, comp);

  std::cout << "test(" << size << ") done" << std::endl;
}


int main(int argc, char** argv)
{
  std::vector<int> sizes;
  sizes.push_back(0);
  sizes.push_back(1);
  sizes.push_back(10);
  sizes.push_back(100);
  sizes.push_back(10000);
  sizes.push_back(25000);
  sizes.push_back(50000);
  sizes.push_back(100000);
  sizes.push_back(250000);
  sizes.push_back(500000);
  sizes.push_back(1000000);
  sizes.push_back(1048576);
  sizes.push_back(2500000);
  sizes.push_back(5000000);
  sizes.push_back(10000000);
  sizes.push_back(50000000);
  sizes.push_back(75000000);

  for(std::vector<int>::iterator i = sizes.begin();
      i != sizes.end();
      ++i)
  {
    test<int>(*i);
  }

  return 0;
}

