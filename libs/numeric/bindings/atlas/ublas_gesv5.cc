
// solving A * X = B
// using driver function gesv()
// with c_vector<> & c_matrix<> 

#include <cstddef>
#include <iostream>
#include <boost/numeric/bindings/lapack/driver/gesv.hpp>
#include <boost/numeric/bindings/ublas/matrix.hpp>
#include <boost/numeric/bindings/ublas/matrix_proxy.hpp>
#include <boost/numeric/bindings/ublas/vector.hpp>
#include <boost/numeric/bindings/std/vector.hpp>
#include <boost/numeric/ublas/io.hpp> 

namespace ublas = boost::numeric::ublas;
namespace bindings = boost::numeric::bindings;
namespace lapack = boost::numeric::bindings::lapack;

using std::size_t; 
using std::cout;
using std::endl; 

typedef ublas::c_matrix<double, 4, 4> m4x4_t;
typedef ublas::c_matrix<double, 3, 3> m3x3_t;
typedef ublas::c_matrix<double, 1, 3> mrhs_t;
typedef ublas::c_vector<double, 5> v5_t; 

int main() {

  cout << endl; 
  size_t n = 3;

  m4x4_t a (n, n);   // system matrix 
  a(0,0) = 1.; a(0,1) = 1.; a(0,2) = 1.;
  a(1,0) = 2.; a(1,1) = 3.; a(1,2) = 1.;
  a(2,0) = 1.; a(2,1) = -1.; a(2,2) = -1.;

  mrhs_t b (1, n);  // right-hand side vector
  b(0,0) = 4.; 
  b(0,1) = 9.;
  b(0,2) = -2.; 

#ifndef BOOST_NUMERIC_BINDINGS_POOR_MANS_TRAITS
  m3x3_t a2; // for part 2
  a2 = project (a, ublas::range (0,3), ublas::range (0,3)); 
  v5_t b2 (n); 
  b2 = row (b, 0); 
#endif 

  // part 1:
  cout << "A: " << a << endl; 
  cout << "B: " << b << endl; 

  std::vector< int > pivota( bindings::size1( a ) );
  lapack::gesv (a, pivota, b);  
  cout << "X: " << b << endl; 

  cout << endl; 

#ifndef BOOST_NUMERIC_BINDINGS_POOR_MANS_TRAITS
  // part 2:
  cout << "A: " << a2 << endl; 
  cout << "B: " << b2 << endl; 

  std::vector< int > pivota2( bindings::size1( a2 ) );
  lapack::gesv (a2, pivota2, b2);  
  cout << "X: " << b2 << endl; 

  cout << endl; 
#endif
}

