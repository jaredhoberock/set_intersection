#pragma once

#include "set_operation.h"

namespace set_intersection_detail
{


using thrust::detail::uint32_t;


struct serial_bounded_set_intersection
{
  // max_input_size <= 32
  template<typename Size, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Compare>
  inline __device__
    uint32_t operator()(Size max_input_size,
                        InputIterator1 first1, InputIterator1 last1,
                        InputIterator2 first2, InputIterator2 last2,
                        OutputIterator result,
                        Compare comp)
  {
    uint32_t active_mask = 0;
    uint32_t active_bit = 1;
  
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


  template<typename Size, typename InputIterator1, typename InputIterator2, typename Compare>
  inline __device__
    Size count(Size max_input_size,
               InputIterator1 first1, InputIterator1 last1,
               InputIterator2 first2, InputIterator2 last2,
               Compare comp)
  {
    Size result = 0;
  
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
}; // end serial_bounded_set_intersection


} // end set_intersection_detail


template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Compare>
  OutputIterator my_set_intersection(InputIterator1 first1, InputIterator1 last1,
                                     InputIterator2 first2, InputIterator2 last2,
                                     OutputIterator result,
                                     Compare comp)
{
  return set_operation(first1, last1, first2, last2, result, comp, set_intersection_detail::serial_bounded_set_intersection());
}

