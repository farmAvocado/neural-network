# coding: utf8
# cython: boundscheck=True
# cython: wraparound=False
# cython: nonecheck=True

import numpy as np
from scipy.linalg.cython_blas cimport dgemm, dgemv
from libc.stdlib cimport malloc, free
from libc.string cimport memset, memcpy

ctypedef double float_t
DEF FLOAT_T = 'double'

# Fortan's gemm function computes 
#   A * B = C^T (1)
# the C^T is due to Fortran data ordering.
# So here need to compute 
#   B^T * A^T = C (2)
# Since all A, B, C are C-contiguous (i.e. the traonsposation of Fortran-contiguous one),
# we need to tell gemm that before multiply A and B, transpose them (i.e. transa = transb = 'T')
cpdef void rmo_dgemm(float_t[:,:] A, bint ta, float_t[:,:] B, bint tb, float_t[:,:] C, float_t alpha=1.0, float_t beta=0.0) nogil:
  cdef:
    char transa = 'N' if ta == 0 else 'T'
    char transb = 'N' if tb == 0 else 'T'
    int m = C.shape[1]
    int n = C.shape[0]
    int k = B.shape[0] if transb == 'N' else B.shape[1]
    int lda = A.shape[1]
    int ldb = B.shape[1]
    int ldc = C.shape[1]

  dgemm(&transb, &transa,
        &m, &n, &k, &alpha,
        &B[0,0], &ldb,
        &A[0,0], &lda,
        &beta,
        &C[0,0], &ldc)

cpdef void rmo_dgemv(float_t[:,:] A, bint ta, float_t[:] x, float_t[:] y, float_t alpha=1.0, float_t beta=0.0) :
  cdef:
    char transa = 'N' if ta == 1 else 'T'
    int m = A.shape[1]
    int n = A.shape[0]
    int lda = A.shape[1]
    int incx = 1
    int incy = 1

  dgemv(&transa,
        &m, &n, &alpha,
        &A[0,0], &lda,
        &x[0], &incx,
        &beta,
        &y[0], &incy)


# initialization
############################################################
RNG = np.random.RandomState(17)

cpdef init_uniform(float_t minx, float_t maxx, tuple shape):
  return RNG.uniform(minx, maxx, shape).astype(FLOAT_T)


# connection layer
############################################################
cdef float_t[:] ONES = np.ones(5000, dtype=FLOAT_T)

cdef class Dense:
  cdef:
    public bint has_param
    public float_t[:,:] W
    public float_t[:] b
    public float_t[:,:] W_grad
    public float_t[:] b_grad
    public float_t[:,:] inp
    public float_t[:,:] outp
    public int n_in, n_out
    public float_t[:,:] err

  def __cinit__(self, n_in, n_out):
    self.W = init_uniform(0, 1.0, (n_out, n_in))
    self.b = np.zeros(n_out, dtype=FLOAT_T)
    self.W_grad = np.empty_like(self.W, dtype=FLOAT_T)
    self.b_grad = np.empty_like(self.b, dtype=FLOAT_T)
    self.inp = None
    self.outp = None
    self.n_in = n_in
    self.n_out = n_out
    self.err = None

  def __dealloc__(self):
    if self.outp is not None:
      free(&self.outp[0,0])
      self.outp = None
      free(&self.err[0,0])
      self.err = None

  cpdef void forward(self, float_t[:,:] X):
    cdef:
      int m = X.shape[0]
      int n = self.n_out
      int k = self.n_in

    self.inp = X
    if self.outp is None:
      self.outp = <float_t[:m,:n]>malloc(sizeof(float_t) * m * n)
      self.err = <float_t[:m,:k]>malloc(sizeof(float_t) * m * k)
    elif self.outp.shape[0] < m:
      free(&self.outp[0,0])
      self.outp = <float_t[:m,:n]>malloc(sizeof(float_t) * m * n)
      free(&self.err[0,0])
      self.err = <float_t[:m,:k]>malloc(sizeof(float_t) * m * k)

    rmo_dgemm(X, 0, self.W, 1, self.outp)
    # leverage blas, no manual broadcast function
    rmo_dgemm(<float_t[:1,:m]>&ONES[0], 1, 
        <float_t[:1,:n]>&self.b[0], 0, self.outp, beta=1.0)

  cpdef void backward(self, float_t[:,:] X):
    cdef:
      int m = X.shape[0]

    rmo_dgemm(X, 1, self.inp, 0, self.W_grad)
    rmo_dgemv(X, 0, <float_t[:m]>&ONES[0], self.b_grad)
    rmo_dgemm(X, 0, self.W, 0, self.err)
