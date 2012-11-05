#pragma once

#include "set_operation.h"

namespace set_symmetric_difference_detail
{


using thrust::detail::uint32_t;


struct serial_bounded_set_symmetric_difference
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
        *result = *first1;
        active_mask |= active_bit;
        ++first1;
      } // end if
      else if(comp(*first2,*first1))
      {
        *result = *first2;
        active_mask |= active_bit;
        ++first2;
      } // end else if
      else
      {
        ++first1;
        ++first2;
      } // end else
  
      ++result;
      active_bit <<= 1;
    } // end while

    while(first1 != last1)
    {
      *result = *first1;
      ++first1;
      ++result;
      active_mask |= active_bit;
      active_bit <<= 1;
    }

    while(first2 != last2)
    {
      *result = *first2;
      ++first2;
      ++result;
      active_mask |= active_bit;
      active_bit <<= 1;
    }
  
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
        ++result;
      } // end if
      else if(comp(*first2,*first1))
      {
        ++first2;
        ++result;
      } // end else if
      else
      {
        ++first1;
        ++first2;
      } // end else
    } // end while
  
    return result + thrust::max(last1 - first1,last2 - first2);
  }
}; // end serial_bounded_set_symmetric_difference


} // end set_symmetric_difference_detail


template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Compare>
  OutputIterator my_set_symmetric_difference(InputIterator1 first1, InputIterator1 last1,
                                     InputIterator2 first2, InputIterator2 last2,
                                     OutputIterator result,
                                     Compare comp)
{
  return set_operation(first1, last1, first2, last2, result, comp, set_symmetric_difference_detail::serial_bounded_set_symmetric_difference());
}
