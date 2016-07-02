# coding: utf8
# cython: boundscheck=True
# cython: wraparound=False
# cython: nonecheck=True

from scipy.linalg.cython_blas cimport sgemm

# Fortan's gemm function computes 
#   A * B = C^T (1)
# the C^T is due to Fortran data ordering.
# So here need to compute 
#   B^T * A^T = C (2)
# Since all A, B, C are C-contiguous (i.e. the traonsposation of Fortran-contiguous one),
# we need to tell gemm that before multiply A and B, transpose them (i.e. transa = transb = 'T')
cpdef void rmo_sgemm(float[:,:] A, bint ta, float[:,:] B, bint tb, float[:,:] C):
  cdef:
    char transa = 'N' if ta == 0 else 'T'
    char transb = 'N' if tb == 0 else 'T'
    int m = C.shape[1]
    int n = C.shape[0]
    int k = B.shape[0] if transb == 'N' else B.shape[1]
    int lda = A.shape[1]
    int ldb = B.shape[1]
    int ldc = C.shape[1]
    float alpha = 1.0
    float beta = 0.0

  print(chr(transb), chr(transa), m, n, k, ldb, lda, ldc)
  sgemm(&transb, &transa,
        &m, &n, &k, &alpha,
        &B[0,0], &ldb,
        &A[0,0], &lda,
        &beta,
        &C[0,0], &ldc)
