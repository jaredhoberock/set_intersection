#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>
#include "utility.h"
#include "set_intersection.h"
#include "time_invocation_cuda.hpp"


namespace my
{

template<typename T>
struct random
{
  template<typename RNG>
  T operator()(RNG rand)
  {
    return rand();
  }
};

}


template<typename T>
void test(int size)
{
  int a_size = size;
  int b_size = size + 1;
  int input_size = a_size + b_size;

  typedef thrust::less<T> Compare;
  Compare comp;

  thrust::device_vector<T> a, b;
  random_sorted_vector(a_size, comp, a, my::random<T>());
  random_sorted_vector(b_size, comp, b, my::random<T>());

  thrust::device_vector<T> o(input_size, T(13));

  o.erase(my_set_intersection(a.begin(), a.end(), b.begin(), b.end(), o.begin(), comp), o.end());

  thrust::host_vector<T> h_a = a, h_b = b;
  thrust::host_vector<T> reference(input_size, T(13));
  reference.erase(std::set_intersection(h_a.begin(), h_a.end(), h_b.begin(), h_b.end(), reference.begin(), comp), reference.end());

  verify(reference, a, b, o);

  std::cout << "test(" << size << ") done" << std::endl;

  typedef typename thrust::device_vector<T>::iterator Iterator;
  double msecs = time_invocation_cuda(20, my_set_intersection<Iterator,Iterator,Iterator,Compare>, a.begin(), a.end(), b.begin(), b.end(), o.begin(), comp);

  std::cout << "  Input size: " << input_size << std::endl;
  std::cout << "  Input ratio: " << float(a_size) / b_size << std::endl;
  std::cout << "  Output size: " << o.size() << std::endl;
  std::cout << "  Time: " << msecs << " ms" << std::endl;
  double seconds = msecs / 1000;
  double megakeys = double(input_size) / (1 << 20);
  double megakeys_per_second = megakeys  / seconds;
  std::cout << "  Throughput: " << megakeys_per_second << " Mkeys/s" << std::endl
            << std::endl;
}

namespace my
{

template<typename T>
struct random<thrust::tuple<T,T> >
{
  template<typename RNG>
  thrust::tuple<T,T> operator()(RNG rand)
  {
    return thrust::make_tuple(rand(),rand());
  }
};


template<typename T>
struct random<thrust::tuple<T,T,T> >
{
  template<typename RNG>
  thrust::tuple<T,T,T> operator()(RNG rand)
  {
    return thrust::make_tuple(rand(),rand(),rand());
  }
};


template<typename T>
struct random<thrust::tuple<T,T,T,T> >
{
  template<typename RNG>
  thrust::tuple<T,T,T,T> operator()(RNG rand)
  {
    return thrust::make_tuple(rand(),rand(),rand(),rand());
  }
};


template<typename T>
struct random<thrust::tuple<T,T,T,T,T> >
{
  template<typename RNG>
  thrust::tuple<T,T,T,T,T> operator()(RNG rand)
  {
    return thrust::make_tuple(rand(),rand(),rand(),rand(),rand());
  }
};

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
    //test<thrust::tuple<double,double,double,double,double> >(*i);
    //test<thrust::tuple<uint64_t,uint64_t,uint64_t,char> >(*i);
  }

  return 0;
}

