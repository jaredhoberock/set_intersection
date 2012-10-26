#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/mismatch.h>
#include <cstdlib>
#include <algorithm>
#include <iostream>

namespace {

int small_rand() {
    return std::rand() & 0xffff;
}

template<typename T, typename Compare>
thrust::host_vector<T> make_random_sorted_vector(int size, Compare comp) {
    thrust::host_vector<T> result(size);
    thrust::generate(result.begin(), result.end(), small_rand);
    thrust::sort(result.begin(), result.end(), comp);
    return result;
}

}


template<typename T, typename Compare>
void random_sorted_vector(int size, Compare comp, thrust::device_vector<T> &vec)
{
  thrust::host_vector<T> h_vec = make_random_sorted_vector<T>(size, comp);
  vec = h_vec;
}

template<typename T, typename Compare>
void random_sorted_vector(int size, Compare comp, thrust::host_vector<T> &vec)
{
  vec = make_random_sorted_vector<T>(size, comp);
}


template<typename T, typename Compare>
void verify(const thrust::host_vector<T> &h_a,
            const thrust::host_vector<T> &h_b,
            const thrust::host_vector<T> &h_o,
            Compare comp)
{
  thrust::host_vector<T> reference(h_a.size() + h_b.size(), T(13));
  reference.erase(std::set_intersection(h_a.begin(), h_a.end(),
                                        h_b.begin(), h_b.end(),
                                        reference.begin(),
                                        comp),
                  reference.end());
  
  for(int i = 0; i < h_o.size(); i++) {
      if (h_o[i] != reference[i]) {
          std::cerr << "    Error at: " << i << ": mine: " << h_o[i] <<
              ", std::set_intersection: " << reference[i] << std::endl;
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


template<typename T, typename Compare>
void verify(const thrust::device_vector<T> &a,
            const thrust::device_vector<T> &b,
            const thrust::device_vector<T> &o,
            Compare comp)
{
    thrust::host_vector<T> h_a = a;
    thrust::host_vector<T> h_b = b;
    thrust::host_vector<T> h_o = o;

    verify(h_a, h_b, h_o, comp);
}

