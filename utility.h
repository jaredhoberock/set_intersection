#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/mismatch.h>
#include <thrust/functional.h>
#include <cstdlib>
#include <algorithm>
#include <iostream>

namespace {

int small_rand() {
    return std::rand() & 0xffff;
}

template<typename T, typename Compare, typename Function>
thrust::host_vector<T> make_random_sorted_vector(int size, Compare comp, Function f) {
    thrust::host_vector<T> result(size);
    for(unsigned int i = 0; i < result.size(); ++i)
    {
      result[i] = f(small_rand);
    }
    thrust::sort(result.begin(), result.end(), comp);
    return result;
}


template<typename T>
struct passthrough_rng
{
  template<typename RNG>
  T operator()(RNG rand)
  {
    return rand();
  }
};


template<typename T, typename Compare>
thrust::host_vector<T> make_random_sorted_vector(int size, Compare comp)
{
  return make_random_sorted_vector<T>(size, comp, passthrough_rng<T>());
}

}


template<typename T, typename Compare, typename Function>
void random_sorted_vector(int size, Compare comp, thrust::device_vector<T> &vec, Function f)
{
  thrust::host_vector<T> h_vec = make_random_sorted_vector<T>(size, comp, f);
  vec = h_vec;
}

template<typename T, typename Compare, typename Function>
void random_sorted_vector(int size, Compare comp, thrust::host_vector<T> &vec, Function f)
{
  vec = make_random_sorted_vector<T>(size, comp);
}


template<typename T>
void verify(const thrust::host_vector<T> &reference,
            const thrust::host_vector<T> &h_a,
            const thrust::host_vector<T> &h_b,
            const thrust::host_vector<T> &h_o)
{
  for(int i = 0; i < h_o.size(); i++) {
      if (h_o[i] != reference[i]) {
          std::cerr << "    Error at: " << i << ": mine: " << h_o[i] <<
              ", reference: " << reference[i] << std::endl;
      }
  }
  
  if(h_o.size() < reference.size())
  {
    std::cerr << "    Error: Missing data." << std::endl;
  }

  if(h_o.size() != reference.size())
  {
    std::cerr << "    Error: Wrong size." << std::endl;
  }
}


template<typename T>
void verify(const thrust::host_vector<T> &reference,
            const thrust::device_vector<T> &a,
            const thrust::device_vector<T> &b,
            const thrust::device_vector<T> &o)
{
    thrust::host_vector<T> h_a = a;
    thrust::host_vector<T> h_b = b;
    thrust::host_vector<T> h_o = o;

    verify(reference, h_a, h_b, h_o);
}

