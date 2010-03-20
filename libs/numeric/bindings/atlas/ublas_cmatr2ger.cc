
// BLAS level 2 -- complex numbers

//#define BOOST_NO_FUNCTION_TEMPLATE_ORDERING

#include <iostream>
#include <complex>
#include <boost/numeric/bindings/blas/level1.hpp>
#include <boost/numeric/bindings/blas/level2.hpp>
#include <boost/numeric/bindings/ublas/vector.hpp>
#include <boost/numeric/bindings/ublas/matrix.hpp>
#ifdef F_USE_STD_VECTOR
#include <vector>
#include <boost/numeric/bindings/std/vector.hpp> 
#endif 
#include "utils.h" 

namespace ublas = boost::numeric::ublas;
namespace blas = boost::numeric::bindings::blas;

using std::cout;
using std::endl; 

typedef double real_t;
typedef std::complex<real_t> cmplx_t; 

#ifndef F_USE_STD_VECTOR
typedef ublas::vector<cmplx_t> vct_t;
typedef ublas::matrix<cmplx_t, ublas::row_major> m_t;
#else
typedef ublas::vector<cmplx_t, std::vector<cmplx_t> > vct_t;
typedef ublas::matrix<cmplx_t, ublas::column_major, std::vector<cmplx_t> > m_t;
#endif 

int main() {

  cout << endl; 

  vct_t vx (2);
  vct_t vy (4); // vector size can be larger 
                // than corresponding matrix size 
  blas::set (cmplx_t (1., 0.), vx);
  blas::set (cmplx_t (0., 2.), vy); 
  print_v (vx, "vx"); 
  cout << endl; 
  print_v (vy, "vy"); 
  cout << endl; 

  m_t m (2, 3);
  init_m (m, const_val<cmplx_t>()); 
  print_m (m, "m"); 
  cout << endl; 

  // m += x y^T
  blas::gerc (1.0, vx, vy, m);   // bindings extension 
  print_m (m, "m += x y^T"); 
  cout << endl << endl; 

  init_m (m, const_val<cmplx_t>()); 
  print_m (m, "m"); 
  cout << endl; 

  // m += x y^T
  blas::geru (cmplx_t (1.0, 0.0), vx, vy, m); 
  print_m (m, "m += x y^T"); 
  cout << endl << endl; 

  m (0, 0) = cmplx_t (0., 1.);
  m (0, 1) = cmplx_t (0., 2.);
  m (0, 2) = cmplx_t (0., 3.);
  m (1, 0) = cmplx_t (0., 4.);
  m (1, 1) = cmplx_t (0., 5.);
  m (1, 2) = cmplx_t (0., 6.);
  print_m (m, "m"); 
  cout << endl; 

  // m += 2 x y^T
  blas::geru (cmplx_t (2., 0.), vx, vy, m);  
  print_m (m, "m += 2 x y^T"); 
  cout << endl << endl; 

  m (0, 0) = cmplx_t (-1., 1.);
  m (0, 1) = cmplx_t (-2., 2.);
  m (0, 2) = cmplx_t (-3., 3.);
  m (1, 0) = cmplx_t (-4., 4.);
  m (1, 1) = cmplx_t (-5., 5.);
  m (1, 2) = cmplx_t (-6., 6.);
  print_m (m, "m"); 
  cout << endl; 

  // m += x y^H
  blas::gerc (cmplx_t (1., 0.), vx, vy, m);   // bindings extension 
  print_m (m, "m += x y^H"); 
  cout << endl << endl; 

}
