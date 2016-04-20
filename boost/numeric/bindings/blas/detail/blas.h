//
// Copyright (c) 2003--2009
// Toon Knapen, Karl Meerbergen, Kresimir Fresl,
// Thomas Klimpel and Rutger ter Borg
//
// Copyright (c) 2016
// Heiko Bauke
// 
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
// THIS FILE IS AUTOMATICALLY GENERATED
// PLEASE DO NOT EDIT!
//

#ifndef BOOST_NUMERIC_BINDINGS_BLAS_DETAIL_BLAS_H
#define BOOST_NUMERIC_BINDINGS_BLAS_DETAIL_BLAS_H

#include <boost/numeric/bindings/blas/detail/blas_names.h>

extern "C" {

  //
  // BLAS level1 routines
  //

  // Value-type variants of asum
  float BLAS_SASUM(const fortran_int_t* n, const float* x,
		   const fortran_int_t* incx);
  double BLAS_DASUM(const fortran_int_t* n, const double* x,
		    const fortran_int_t* incx);
  float BLAS_SCASUM(const fortran_int_t* n, const void* x,
		    const fortran_int_t* incx);
  double BLAS_DZASUM(const fortran_int_t* n, const void* x,
		     const fortran_int_t* incx);

  // Value-type variants of axpy
  void BLAS_SAXPY(const fortran_int_t* n, const float* a, const float* x,
		  const fortran_int_t* incx, float* y, const fortran_int_t* incy);
  void BLAS_DAXPY(const fortran_int_t* n, const double* a, const double* x,
		  const fortran_int_t* incx, double* y, const fortran_int_t* incy);
  void BLAS_CAXPY(const fortran_int_t* n, const void* a, const void* x,
		  const fortran_int_t* incx, void* y, const fortran_int_t* incy);
  void BLAS_ZAXPY(const fortran_int_t* n, const void* a, const void* x,
		  const fortran_int_t* incx, void* y, const fortran_int_t* incy);

  // Value-type variants of copy
  void BLAS_SCOPY(const fortran_int_t* n, const float* x,
		  const fortran_int_t* incx, float* y, const fortran_int_t* incy);
  void BLAS_DCOPY(const fortran_int_t* n, const double* x,
		  const fortran_int_t* incx, double* y, const fortran_int_t* incy);
  void BLAS_CCOPY(const fortran_int_t* n, const void* x,
		  const fortran_int_t* incx, void* y, const fortran_int_t* incy);
  void BLAS_ZCOPY(const fortran_int_t* n, const void* x,
		  const fortran_int_t* incx, void* y, const fortran_int_t* incy);

  // Value-type variants of dot
  float BLAS_SDOT(const fortran_int_t* n, const float* x,
		  const fortran_int_t* incx, const float* y, const fortran_int_t* incy);
  double BLAS_DDOT(const fortran_int_t* n, const double* x,
		   const fortran_int_t* incx, const double* y,
		   const fortran_int_t* incy);

  // Value-type variants of dotu
#if defined BIND_FORTRAN_RETURN_COMPLEX_FIRST_ARG
  void BLAS_CDOTU(void* res, const fortran_int_t* n, const void* x,
		  const fortran_int_t* incx, const void* y, const fortran_int_t* incy);
  void BLAS_ZDOTU(void* res, const fortran_int_t* n, const void* x,
		  const fortran_int_t* incx, const void* y, const fortran_int_t* incy);
#elif defined BIND_FORTRAN_RETURN_COMPLEX_LAST_ARG
  void BLAS_CDOTU(const fortran_int_t* n, const void* x,
		  const fortran_int_t* incx, const void* y, const fortran_int_t* incy, void* res);
  void BLAS_ZDOTU(const fortran_int_t* n, const void* x,
		  const fortran_int_t* incx, const void* y, const fortran_int_t* incy, void* res);
#else
  boost::numeric::bindings::traits::complex_f BLAS_CDOTU(const fortran_int_t* n, const void* x,
							 const fortran_int_t* incx, const void* y, const fortran_int_t* incy);
  boost::numeric::bindings::traits::complex_d BLAS_ZDOTU(const fortran_int_t* n, const void* x,
							 const fortran_int_t* incx, const void* y, const fortran_int_t* incy);
#endif
  
  // Value-type variants of doth
#if defined BIND_FORTRAN_RETURN_COMPLEX_FIRST_ARG
  void BLAS_CDOTH(void* res, const fortran_int_t* n, const void* x,
		  const fortran_int_t* incx, const void* y, const fortran_int_t* incy);
  void BLAS_ZDOTH(void* res, const fortran_int_t* n, const void* x,
		  const fortran_int_t* incx, const void* y, const fortran_int_t* incy);
#elif defined BIND_FORTRAN_RETURN_COMPLEX_LAST_ARG
  void BLAS_CDOTH(const fortran_int_t* n, const void* x,
		  const fortran_int_t* incx, const void* y, const fortran_int_t* incy, void* res);
  void BLAS_ZDOTH(const fortran_int_t* n, const void* x,
		  const fortran_int_t* incx, const void* y, const fortran_int_t* incy, void* res);
#else
  boost::numeric::bindings::traits::complex_f BLAS_CDOTH(const fortran_int_t* n, const void* x,
							 const fortran_int_t* incx, const void* y, const fortran_int_t* incy);
  boost::numeric::bindings::traits::complex_d BLAS_ZDOTH(const fortran_int_t* n, const void* x,
							 const fortran_int_t* incx, const void* y, const fortran_int_t* incy);
#endif

  // Value-type variants of dotc
  boost::numeric::bindings::traits::complex_f BLAS_CDOTC(const fortran_int_t* n, const void* x,
							 const fortran_int_t* incx, const void* y, const fortran_int_t* incy);
  boost::numeric::bindings::traits::complex_d BLAS_ZDOTC(const fortran_int_t* n, const void* x,
							 const fortran_int_t* incx, const void* y, const fortran_int_t* incy);

  // Value-type variants of iamax
  fortran_int_t BLAS_ISAMAX(const fortran_int_t* n, const float* x,
			    const fortran_int_t* incx);
  fortran_int_t BLAS_IDAMAX(const fortran_int_t* n, const double* x,
			    const fortran_int_t* incx);
  fortran_int_t BLAS_ICAMAX(const fortran_int_t* n, const void* x,
			    const fortran_int_t* incx);
  fortran_int_t BLAS_IZAMAX(const fortran_int_t* n, const void* x,
			    const fortran_int_t* incx);

  // Value-type variants of nrm2
  float BLAS_SNRM2(const fortran_int_t* n, const float* x,
		   const fortran_int_t* incx);
  double BLAS_DNRM2(const fortran_int_t* n, const double* x,
		    const fortran_int_t* incx);
  float BLAS_SCNRM2(const fortran_int_t* n, const void* x,
		    const fortran_int_t* incx);
  double BLAS_DZNRM2(const fortran_int_t* n, const void* x,
		     const fortran_int_t* incx);

  // Value-type variants of prec_dot
  double BLAS_DSDOT(const fortran_int_t* n, const float* x,
		    const fortran_int_t* incx, const float* y, const fortran_int_t* incy);

  // Value-type variants of rot
  void BLAS_SROT(const fortran_int_t* n, float* x, const fortran_int_t* incx,
		 float* y, const fortran_int_t* incy, const float* c, const float* s);
  void BLAS_DROT(const fortran_int_t* n, double* x, const fortran_int_t* incx,
		 double* y, const fortran_int_t* incy, const double* c,
		 const double* s);
  void BLAS_CSROT(const fortran_int_t* n, void* x, const fortran_int_t* incx,
		  void* y, const fortran_int_t* incy, const float* c, const float* s);
  void BLAS_ZDROT(const fortran_int_t* n, void* x, const fortran_int_t* incx,
		  void* y, const fortran_int_t* incy, const double* c, const double* s);

  // Value-type variants of rotg
  void BLAS_SROTG(float* a, float* b, float* c, float* s);
  void BLAS_DROTG(double* a, double* b, double* c, double* s);
  void BLAS_CROTG(void* a, void* b, float* c, void* s);
  void BLAS_ZROTG(void* a, void* b, double* c, void* s);

  // Value-type variants of rotm
  void BLAS_SROTM(const fortran_int_t* n, float* x, const fortran_int_t* incx,
		  float* y, const fortran_int_t* incy, float* param);
  void BLAS_DROTM(const fortran_int_t* n, double* x, const fortran_int_t* incx,
		  double* y, const fortran_int_t* incy, double* param);

  // Value-type variants of rotmg
  void BLAS_SROTMG(float* d1, float* d2, float* x1, const float* y1,
		   float* sparam);
  void BLAS_DROTMG(double* d1, double* d2, double* x1, const double* y1,
		   double* dparam);

  // Value-type variants of scal
  void BLAS_SSCAL(const fortran_int_t* n, const float* a, float* x,
		  const fortran_int_t* incx);
  void BLAS_DSCAL(const fortran_int_t* n, const double* a, double* x,
		  const fortran_int_t* incx);
  void BLAS_CSSCAL(const fortran_int_t* n, const float* a, void* x,
		   const fortran_int_t* incx);
  void BLAS_ZDSCAL(const fortran_int_t* n, const double* a, void* x,
		   const fortran_int_t* incx);
  void BLAS_CSCAL(const fortran_int_t* n, const void* a, void* x,
		  const fortran_int_t* incx);
  void BLAS_ZSCAL(const fortran_int_t* n, const void* a, void* x,
		  const fortran_int_t* incx);

  // Value-type variants of swap
  void BLAS_SSWAP(const fortran_int_t* n, float* x, const fortran_int_t* incx,
		  float* y, const fortran_int_t* incy);
  void BLAS_DSWAP(const fortran_int_t* n, double* x, const fortran_int_t* incx,
		  double* y, const fortran_int_t* incy);
  void BLAS_CSWAP(const fortran_int_t* n, void* x, const fortran_int_t* incx,
		  void* y, const fortran_int_t* incy);
  void BLAS_ZSWAP(const fortran_int_t* n, void* x, const fortran_int_t* incx,
		  void* y, const fortran_int_t* incy);

  //
  // BLAS level2 routines
  //

  // Value-type variants of gbmv
  void BLAS_SGBMV(const char* trans, const fortran_int_t* m,
		  const fortran_int_t* n, const fortran_int_t* kl,
		  const fortran_int_t* ku, const float* alpha, const float* a,
		  const fortran_int_t* lda, const float* x, const fortran_int_t* incx,
		  const float* beta, float* y, const fortran_int_t* incy);
  void BLAS_DGBMV(const char* trans, const fortran_int_t* m,
		  const fortran_int_t* n, const fortran_int_t* kl,
		  const fortran_int_t* ku, const double* alpha, const double* a,
		  const fortran_int_t* lda, const double* x, const fortran_int_t* incx,
		  const double* beta, double* y, const fortran_int_t* incy);
  void BLAS_CGBMV(const char* trans, const fortran_int_t* m,
		  const fortran_int_t* n, const fortran_int_t* kl,
		  const fortran_int_t* ku, const void* alpha, const void* a,
		  const fortran_int_t* lda, const void* x, const fortran_int_t* incx,
		  const void* beta, void* y, const fortran_int_t* incy);
  void BLAS_ZGBMV(const char* trans, const fortran_int_t* m,
		  const fortran_int_t* n, const fortran_int_t* kl,
		  const fortran_int_t* ku, const void* alpha, const void* a,
		  const fortran_int_t* lda, const void* x, const fortran_int_t* incx,
		  const void* beta, void* y, const fortran_int_t* incy);

  // Value-type variants of gemv
  void BLAS_SGEMV(const char* trans, const fortran_int_t* m,
		  const fortran_int_t* n, const float* alpha, const float* a,
		  const fortran_int_t* lda, const float* x, const fortran_int_t* incx,
		  const float* beta, float* y, const fortran_int_t* incy);
  void BLAS_DGEMV(const char* trans, const fortran_int_t* m,
		  const fortran_int_t* n, const double* alpha, const double* a,
		  const fortran_int_t* lda, const double* x, const fortran_int_t* incx,
		  const double* beta, double* y, const fortran_int_t* incy);
  void BLAS_CGEMV(const char* trans, const fortran_int_t* m,
		  const fortran_int_t* n, const void* alpha, const void* a,
		  const fortran_int_t* lda, const void* x, const fortran_int_t* incx,
		  const void* beta, void* y, const fortran_int_t* incy);
  void BLAS_ZGEMV(const char* trans, const fortran_int_t* m,
		  const fortran_int_t* n, const void* alpha, const void* a,
		  const fortran_int_t* lda, const void* x, const fortran_int_t* incx,
		  const void* beta, void* y, const fortran_int_t* incy);

  // Value-type variants of ger
  void BLAS_SGER(const fortran_int_t* m, const fortran_int_t* n,
		 const float* alpha, const float* x, const fortran_int_t* incx,
		 const float* y, const fortran_int_t* incy, float* a,
		 const fortran_int_t* lda);
  void BLAS_DGER(const fortran_int_t* m, const fortran_int_t* n,
		 const double* alpha, const double* x, const fortran_int_t* incx,
		 const double* y, const fortran_int_t* incy, double* a,
		 const fortran_int_t* lda);

  // Value-type variants of gerc
  void BLAS_CGERC(const fortran_int_t* m, const fortran_int_t* n,
		  const void* alpha, const void* x, const fortran_int_t* incx,
		  const void* y, const fortran_int_t* incy, void* a,
		  const fortran_int_t* lda);
  void BLAS_ZGERC(const fortran_int_t* m, const fortran_int_t* n,
		  const void* alpha, const void* x, const fortran_int_t* incx,
		  const void* y, const fortran_int_t* incy, void* a,
		  const fortran_int_t* lda);

  // Value-type variants of geru
  void BLAS_CGERU(const fortran_int_t* m, const fortran_int_t* n,
		  const void* alpha, const void* x, const fortran_int_t* incx,
		  const void* y, const fortran_int_t* incy, void* a,
		  const fortran_int_t* lda);
  void BLAS_ZGERU(const fortran_int_t* m, const fortran_int_t* n,
		  const void* alpha, const void* x, const fortran_int_t* incx,
		  const void* y, const fortran_int_t* incy, void* a,
		  const fortran_int_t* lda);

  // Value-type variants of hbmv
  void BLAS_CHBMV(const char* uplo, const fortran_int_t* n,
		  const fortran_int_t* k, const void* alpha, const void* a,
		  const fortran_int_t* lda, const void* x, const fortran_int_t* incx,
		  const void* beta, void* y, const fortran_int_t* incy);
  void BLAS_ZHBMV(const char* uplo, const fortran_int_t* n,
		  const fortran_int_t* k, const void* alpha, const void* a,
		  const fortran_int_t* lda, const void* x, const fortran_int_t* incx,
		  const void* beta, void* y, const fortran_int_t* incy);

  // Value-type variants of hemv
  void BLAS_CHEMV(const char* uplo, const fortran_int_t* n, const void* alpha,
		  const void* a, const fortran_int_t* lda, const void* x,
		  const fortran_int_t* incx, const void* beta, void* y,
		  const fortran_int_t* incy);
  void BLAS_ZHEMV(const char* uplo, const fortran_int_t* n, const void* alpha,
		  const void* a, const fortran_int_t* lda, const void* x,
		  const fortran_int_t* incx, const void* beta, void* y,
		  const fortran_int_t* incy);

  // Value-type variants of her
  void BLAS_CHER(const char* uplo, const fortran_int_t* n, const float* alpha,
		 const void* x, const fortran_int_t* incx, void* a,
		 const fortran_int_t* lda);
  void BLAS_ZHER(const char* uplo, const fortran_int_t* n, const double* alpha,
		 const void* x, const fortran_int_t* incx, void* a,
		 const fortran_int_t* lda);

  // Value-type variants of her2
  void BLAS_CHER2(const char* uplo, const fortran_int_t* n, const void* alpha,
		  const void* x, const fortran_int_t* incx, const void* y,
		  const fortran_int_t* incy, void* a, const fortran_int_t* lda);
  void BLAS_ZHER2(const char* uplo, const fortran_int_t* n, const void* alpha,
		  const void* x, const fortran_int_t* incx, const void* y,
		  const fortran_int_t* incy, void* a, const fortran_int_t* lda);

  // Value-type variants of hpmv
  void BLAS_CHPMV(const char* uplo, const fortran_int_t* n, const void* alpha,
		  const void* ap, const void* x, const fortran_int_t* incx,
		  const void* beta, void* y, const fortran_int_t* incy);
  void BLAS_ZHPMV(const char* uplo, const fortran_int_t* n, const void* alpha,
		  const void* ap, const void* x, const fortran_int_t* incx,
		  const void* beta, void* y, const fortran_int_t* incy);

  // Value-type variants of hpr
  void BLAS_CHPR(const char* uplo, const fortran_int_t* n, const float* alpha,
		 const void* x, const fortran_int_t* incx, void* ap);
  void BLAS_ZHPR(const char* uplo, const fortran_int_t* n, const double* alpha,
		 const void* x, const fortran_int_t* incx, void* ap);

  // Value-type variants of hpr2
  void BLAS_CHPR2(const char* uplo, const fortran_int_t* n, const void* alpha,
		  const void* x, const fortran_int_t* incx, const void* y,
		  const fortran_int_t* incy, void* ap);
  void BLAS_ZHPR2(const char* uplo, const fortran_int_t* n, const void* alpha,
		  const void* x, const fortran_int_t* incx, const void* y,
		  const fortran_int_t* incy, void* ap);

  // Value-type variants of sbmv
  void BLAS_SSBMV(const char* uplo, const fortran_int_t* n,
		  const fortran_int_t* k, const float* alpha, const float* a,
		  const fortran_int_t* lda, const float* x, const fortran_int_t* incx,
		  const float* beta, float* y, const fortran_int_t* incy);
  void BLAS_DSBMV(const char* uplo, const fortran_int_t* n,
		  const fortran_int_t* k, const double* alpha, const double* a,
		  const fortran_int_t* lda, const double* x, const fortran_int_t* incx,
		  const double* beta, double* y, const fortran_int_t* incy);

  // Value-type variants of spmv
  void BLAS_SSPMV(const char* uplo, const fortran_int_t* n, const float* alpha,
		  const float* ap, const float* x, const fortran_int_t* incx,
		  const float* beta, float* y, const fortran_int_t* incy);
  void BLAS_DSPMV(const char* uplo, const fortran_int_t* n,
		  const double* alpha, const double* ap, const double* x,
		  const fortran_int_t* incx, const double* beta, double* y,
		  const fortran_int_t* incy);

  // Value-type variants of spr
  void BLAS_SSPR(const char* uplo, const fortran_int_t* n, const float* alpha,
		 const float* x, const fortran_int_t* incx, float* ap);
  void BLAS_DSPR(const char* uplo, const fortran_int_t* n, const double* alpha,
		 const double* x, const fortran_int_t* incx, double* ap);

  // Value-type variants of spr2
  void BLAS_SSPR2(const char* uplo, const fortran_int_t* n, const float* alpha,
		  const float* x, const fortran_int_t* incx, const float* y,
		  const fortran_int_t* incy, float* ap);
  void BLAS_DSPR2(const char* uplo, const fortran_int_t* n,
		  const double* alpha, const double* x, const fortran_int_t* incx,
		  const double* y, const fortran_int_t* incy, double* ap);

  // Value-type variants of symv
  void BLAS_SSYMV(const char* uplo, const fortran_int_t* n, const float* alpha,
		  const float* a, const fortran_int_t* lda, const float* x,
		  const fortran_int_t* incx, const float* beta, float* y,
		  const fortran_int_t* incy);
  void BLAS_DSYMV(const char* uplo, const fortran_int_t* n,
		  const double* alpha, const double* a, const fortran_int_t* lda,
		  const double* x, const fortran_int_t* incx, const double* beta,
		  double* y, const fortran_int_t* incy);

  // Value-type variants of syr
  void BLAS_SSYR(const char* uplo, const fortran_int_t* n, const float* alpha,
		 const float* x, const fortran_int_t* incx, float* a,
		 const fortran_int_t* lda);
  void BLAS_DSYR(const char* uplo, const fortran_int_t* n, const double* alpha,
		 const double* x, const fortran_int_t* incx, double* a,
		 const fortran_int_t* lda);

  // Value-type variants of syr2
  void BLAS_SSYR2(const char* uplo, const fortran_int_t* n, const float* alpha,
		  const float* x, const fortran_int_t* incx, const float* y,
		  const fortran_int_t* incy, float* a, const fortran_int_t* lda);
  void BLAS_DSYR2(const char* uplo, const fortran_int_t* n,
		  const double* alpha, const double* x, const fortran_int_t* incx,
		  const double* y, const fortran_int_t* incy, double* a,
		  const fortran_int_t* lda);

  // Value-type variants of tbmv
  void BLAS_STBMV(const char* uplo, const char* trans, const char* diag,
		  const fortran_int_t* n, const fortran_int_t* k, const float* a,
		  const fortran_int_t* lda, float* x, const fortran_int_t* incx);
  void BLAS_DTBMV(const char* uplo, const char* trans, const char* diag,
		  const fortran_int_t* n, const fortran_int_t* k, const double* a,
		  const fortran_int_t* lda, double* x, const fortran_int_t* incx);
  void BLAS_CTBMV(const char* uplo, const char* trans, const char* diag,
		  const fortran_int_t* n, const fortran_int_t* k, const void* a,
		  const fortran_int_t* lda, void* x, const fortran_int_t* incx);
  void BLAS_ZTBMV(const char* uplo, const char* trans, const char* diag,
		  const fortran_int_t* n, const fortran_int_t* k, const void* a,
		  const fortran_int_t* lda, void* x, const fortran_int_t* incx);

  // Value-type variants of tbsv
  void BLAS_STBSV(const char* uplo, const char* trans, const char* diag,
		  const fortran_int_t* n, const fortran_int_t* k, const float* a,
		  const fortran_int_t* lda, float* x, const fortran_int_t* incx);
  void BLAS_DTBSV(const char* uplo, const char* trans, const char* diag,
		  const fortran_int_t* n, const fortran_int_t* k, const double* a,
		  const fortran_int_t* lda, double* x, const fortran_int_t* incx);
  void BLAS_CTBSV(const char* uplo, const char* trans, const char* diag,
		  const fortran_int_t* n, const fortran_int_t* k, const void* a,
		  const fortran_int_t* lda, void* x, const fortran_int_t* incx);
  void BLAS_ZTBSV(const char* uplo, const char* trans, const char* diag,
		  const fortran_int_t* n, const fortran_int_t* k, const void* a,
		  const fortran_int_t* lda, void* x, const fortran_int_t* incx);

  // Value-type variants of tpmv
  void BLAS_STPMV(const char* uplo, const char* trans, const char* diag,
		  const fortran_int_t* n, const float* ap, float* x,
		  const fortran_int_t* incx);
  void BLAS_DTPMV(const char* uplo, const char* trans, const char* diag,
		  const fortran_int_t* n, const double* ap, double* x,
		  const fortran_int_t* incx);
  void BLAS_CTPMV(const char* uplo, const char* trans, const char* diag,
		  const fortran_int_t* n, const void* ap, void* x,
		  const fortran_int_t* incx);
  void BLAS_ZTPMV(const char* uplo, const char* trans, const char* diag,
		  const fortran_int_t* n, const void* ap, void* x,
		  const fortran_int_t* incx);

  // Value-type variants of tpsv
  void BLAS_STPSV(const char* uplo, const char* trans, const char* diag,
		  const fortran_int_t* n, const float* ap, float* x,
		  const fortran_int_t* incx);
  void BLAS_DTPSV(const char* uplo, const char* trans, const char* diag,
		  const fortran_int_t* n, const double* ap, double* x,
		  const fortran_int_t* incx);
  void BLAS_CTPSV(const char* uplo, const char* trans, const char* diag,
		  const fortran_int_t* n, const void* ap, void* x,
		  const fortran_int_t* incx);
  void BLAS_ZTPSV(const char* uplo, const char* trans, const char* diag,
		  const fortran_int_t* n, const void* ap, void* x,
		  const fortran_int_t* incx);

  // Value-type variants of trmv
  void BLAS_STRMV(const char* uplo, const char* trans, const char* diag,
		  const fortran_int_t* n, const float* a, const fortran_int_t* lda,
		  float* x, const fortran_int_t* incx);
  void BLAS_DTRMV(const char* uplo, const char* trans, const char* diag,
		  const fortran_int_t* n, const double* a, const fortran_int_t* lda,
		  double* x, const fortran_int_t* incx);
  void BLAS_CTRMV(const char* uplo, const char* trans, const char* diag,
		  const fortran_int_t* n, const void* a, const fortran_int_t* lda,
		  void* x, const fortran_int_t* incx);
  void BLAS_ZTRMV(const char* uplo, const char* trans, const char* diag,
		  const fortran_int_t* n, const void* a, const fortran_int_t* lda,
		  void* x, const fortran_int_t* incx);

  // Value-type variants of trsv
  void BLAS_STRSV(const char* uplo, const char* trans, const char* diag,
		  const fortran_int_t* n, const float* a, const fortran_int_t* lda,
		  float* x, const fortran_int_t* incx);
  void BLAS_DTRSV(const char* uplo, const char* trans, const char* diag,
		  const fortran_int_t* n, const double* a, const fortran_int_t* lda,
		  double* x, const fortran_int_t* incx);
  void BLAS_CTRSV(const char* uplo, const char* trans, const char* diag,
		  const fortran_int_t* n, const void* a, const fortran_int_t* lda,
		  void* x, const fortran_int_t* incx);
  void BLAS_ZTRSV(const char* uplo, const char* trans, const char* diag,
		  const fortran_int_t* n, const void* a, const fortran_int_t* lda,
		  void* x, const fortran_int_t* incx);

  //
  // BLAS level3 routines
  //

  // Value-type variants of gemm
  void BLAS_SGEMM(const char* transa, const char* transb,
		  const fortran_int_t* m, const fortran_int_t* n,
		  const fortran_int_t* k, const float* alpha, const float* a,
		  const fortran_int_t* lda, const float* b, const fortran_int_t* ldb,
		  const float* beta, float* c, const fortran_int_t* ldc);
  void BLAS_DGEMM(const char* transa, const char* transb,
		  const fortran_int_t* m, const fortran_int_t* n,
		  const fortran_int_t* k, const double* alpha, const double* a,
		  const fortran_int_t* lda, const double* b, const fortran_int_t* ldb,
		  const double* beta, double* c, const fortran_int_t* ldc);
  void BLAS_CGEMM(const char* transa, const char* transb,
		  const fortran_int_t* m, const fortran_int_t* n,
		  const fortran_int_t* k, const void* alpha, const void* a,
		  const fortran_int_t* lda, const void* b, const fortran_int_t* ldb,
		  const void* beta, void* c, const fortran_int_t* ldc);
  void BLAS_ZGEMM(const char* transa, const char* transb,
		  const fortran_int_t* m, const fortran_int_t* n,
		  const fortran_int_t* k, const void* alpha, const void* a,
		  const fortran_int_t* lda, const void* b, const fortran_int_t* ldb,
		  const void* beta, void* c, const fortran_int_t* ldc);

  // Value-type variants of hemm
  void BLAS_CHEMM(const char* side, const char* uplo, const fortran_int_t* m,
		  const fortran_int_t* n, const void* alpha, const void* a,
		  const fortran_int_t* lda, const void* b, const fortran_int_t* ldb,
		  const void* beta, void* c, const fortran_int_t* ldc);
  void BLAS_ZHEMM(const char* side, const char* uplo, const fortran_int_t* m,
		  const fortran_int_t* n, const void* alpha, const void* a,
		  const fortran_int_t* lda, const void* b, const fortran_int_t* ldb,
		  const void* beta, void* c, const fortran_int_t* ldc);

  // Value-type variants of her2k
  void BLAS_CHER2K(const char* uplo, const char* trans, const fortran_int_t* n,
		   const fortran_int_t* k, const void* alpha, const void* a,
		   const fortran_int_t* lda, const void* b, const fortran_int_t* ldb,
		   const float* beta, void* c, const fortran_int_t* ldc);
  void BLAS_ZHER2K(const char* uplo, const char* trans, const fortran_int_t* n,
		   const fortran_int_t* k, const void* alpha, const void* a,
		   const fortran_int_t* lda, const void* b, const fortran_int_t* ldb,
		   const double* beta, void* c, const fortran_int_t* ldc);

  // Value-type variants of herk
  void BLAS_CHERK(const char* uplo, const char* trans, const fortran_int_t* n,
		  const fortran_int_t* k, const float* alpha, const void* a,
		  const fortran_int_t* lda, const float* beta, void* c,
		  const fortran_int_t* ldc);
  void BLAS_ZHERK(const char* uplo, const char* trans, const fortran_int_t* n,
		  const fortran_int_t* k, const double* alpha, const void* a,
		  const fortran_int_t* lda, const double* beta, void* c,
		  const fortran_int_t* ldc);

  // Value-type variants of symm
  void BLAS_SSYMM(const char* side, const char* uplo, const fortran_int_t* m,
		  const fortran_int_t* n, const float* alpha, const float* a,
		  const fortran_int_t* lda, const float* b, const fortran_int_t* ldb,
		  const float* beta, float* c, const fortran_int_t* ldc);
  void BLAS_DSYMM(const char* side, const char* uplo, const fortran_int_t* m,
		  const fortran_int_t* n, const double* alpha, const double* a,
		  const fortran_int_t* lda, const double* b, const fortran_int_t* ldb,
		  const double* beta, double* c, const fortran_int_t* ldc);
  void BLAS_CSYMM(const char* side, const char* uplo, const fortran_int_t* m,
		  const fortran_int_t* n, const void* alpha, const void* a,
		  const fortran_int_t* lda, const void* b, const fortran_int_t* ldb,
		  const void* beta, void* c, const fortran_int_t* ldc);
  void BLAS_ZSYMM(const char* side, const char* uplo, const fortran_int_t* m,
		  const fortran_int_t* n, const void* alpha, const void* a,
		  const fortran_int_t* lda, const void* b, const fortran_int_t* ldb,
		  const void* beta, void* c, const fortran_int_t* ldc);

  // Value-type variants of syr2k
  void BLAS_SSYR2K(const char* uplo, const char* trans, const fortran_int_t* n,
		   const fortran_int_t* k, const float* alpha, const float* a,
		   const fortran_int_t* lda, const float* b, const fortran_int_t* ldb,
		   const float* beta, float* c, const fortran_int_t* ldc);
  void BLAS_DSYR2K(const char* uplo, const char* trans, const fortran_int_t* n,
		   const fortran_int_t* k, const double* alpha, const double* a,
		   const fortran_int_t* lda, const double* b, const fortran_int_t* ldb,
		   const double* beta, double* c, const fortran_int_t* ldc);
  void BLAS_CSYR2K(const char* uplo, const char* trans, const fortran_int_t* n,
		   const fortran_int_t* k, const void* alpha, const void* a,
		   const fortran_int_t* lda, const void* b, const fortran_int_t* ldb,
		   const void* beta, void* c, const fortran_int_t* ldc);
  void BLAS_ZSYR2K(const char* uplo, const char* trans, const fortran_int_t* n,
		   const fortran_int_t* k, const void* alpha, const void* a,
		   const fortran_int_t* lda, const void* b, const fortran_int_t* ldb,
		   const void* beta, void* c, const fortran_int_t* ldc);

  // Value-type variants of syrk
  void BLAS_SSYRK(const char* uplo, const char* trans, const fortran_int_t* n,
		  const fortran_int_t* k, const float* alpha, const float* a,
		  const fortran_int_t* lda, const float* beta, float* c,
		  const fortran_int_t* ldc);
  void BLAS_DSYRK(const char* uplo, const char* trans, const fortran_int_t* n,
		  const fortran_int_t* k, const double* alpha, const double* a,
		  const fortran_int_t* lda, const double* beta, double* c,
		  const fortran_int_t* ldc);
  void BLAS_CSYRK(const char* uplo, const char* trans, const fortran_int_t* n,
		  const fortran_int_t* k, const void* alpha, const void* a,
		  const fortran_int_t* lda, const void* beta, void* c,
		  const fortran_int_t* ldc);
  void BLAS_ZSYRK(const char* uplo, const char* trans, const fortran_int_t* n,
		  const fortran_int_t* k, const void* alpha, const void* a,
		  const fortran_int_t* lda, const void* beta, void* c,
		  const fortran_int_t* ldc);

  // Value-type variants of trmm
  void BLAS_STRMM(const char* side, const char* uplo, const char* transa,
		  const char* diag, const fortran_int_t* m, const fortran_int_t* n,
		  const float* alpha, const float* a, const fortran_int_t* lda,
		  float* b, const fortran_int_t* ldb);
  void BLAS_DTRMM(const char* side, const char* uplo, const char* transa,
		  const char* diag, const fortran_int_t* m, const fortran_int_t* n,
		  const double* alpha, const double* a, const fortran_int_t* lda,
		  double* b, const fortran_int_t* ldb);
  void BLAS_CTRMM(const char* side, const char* uplo, const char* transa,
		  const char* diag, const fortran_int_t* m, const fortran_int_t* n,
		  const void* alpha, const void* a, const fortran_int_t* lda, void* b,
		  const fortran_int_t* ldb);
  void BLAS_ZTRMM(const char* side, const char* uplo, const char* transa,
		  const char* diag, const fortran_int_t* m, const fortran_int_t* n,
		  const void* alpha, const void* a, const fortran_int_t* lda, void* b,
		  const fortran_int_t* ldb);

  // Value-type variants of trsm
  void BLAS_STRSM(const char* side, const char* uplo, const char* transa,
		  const char* diag, const fortran_int_t* m, const fortran_int_t* n,
		  const float* alpha, const float* a, const fortran_int_t* lda,
		  float* b, const fortran_int_t* ldb);
  void BLAS_DTRSM(const char* side, const char* uplo, const char* transa,
		  const char* diag, const fortran_int_t* m, const fortran_int_t* n,
		  const double* alpha, const double* a, const fortran_int_t* lda,
		  double* b, const fortran_int_t* ldb);
  void BLAS_CTRSM(const char* side, const char* uplo, const char* transa,
		  const char* diag, const fortran_int_t* m, const fortran_int_t* n,
		  const void* alpha, const void* a, const fortran_int_t* lda, void* b,
		  const fortran_int_t* ldb);
  void BLAS_ZTRSM(const char* side, const char* uplo, const char* transa,
		  const char* diag, const fortran_int_t* m, const fortran_int_t* n,
		  const void* alpha, const void* a, const fortran_int_t* lda, void* b,
		  const fortran_int_t* ldb);

}

#endif
