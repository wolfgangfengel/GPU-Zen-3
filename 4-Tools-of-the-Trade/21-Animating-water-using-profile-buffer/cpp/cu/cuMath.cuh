#ifndef __CU_MATH_CUH_
#define __CU_MATH_CUH_

#include <stdint.h>

#include "math.h"  // CUDA math library
#include <cuda.h>

#include "config_type.h"

// c = {0}
template <int dim>
__forceinline__ __device__ void VecClean(real* c);

template <>
__forceinline__ __device__ void VecClean<2>(real* c) {
	c[0] = c[1] = 0;
}

template <>
__forceinline__ __device__ void VecClean<3>(real* c) {
	c[0] = c[1] = c[2] = 0;
}

// c = a * s
template <int dim>
__forceinline__ __device__ void VecMulVal(const real* a, const real s, real* c);

template <>
__forceinline__ __device__ void VecMulVal<2>(const real* a, const real s, real* c) {
	c[0] = a[0] * s; 
	c[1] = a[1] * s;
}

template <>
__forceinline__ __device__ void VecMulVal<3>(const real* a, const real s, real* c) {
	c[0] = a[0] * s;
	c[1] = a[1] * s;
	c[2] = a[2] * s;
}

// c += a * s
template <int dim>
__forceinline__ __device__ void VecAddonVecMulVal(const real* a, const real s, real* c);

template <>
__forceinline__ __device__ void VecAddonVecMulVal<2>(const real* a, const real s, real* c) {
    c[0] += a[0] * s; 
	c[1] += a[1] * s;
}

template <>
__forceinline__ __device__ void VecAddonVecMulVal<3>(const real* a, const real s, real* c) {
	c[0] += a[0] * s;
	c[1] += a[1] * s;
	c[2] += a[2] * s;
}

// c = a + b * s
template <int dim>
__forceinline__ __device__ void VecAddVecMulVal(const real* a, const real* b, const real s, real* c);

template <>
__forceinline__ __device__ void VecAddVecMulVal<2>(const real* a, const real* b, const real s, real* c) {
	c[0] = a[0] + b[0] * s; 
	c[1] = a[1] + b[1] * s;
}

template <>
__forceinline__ __device__ void VecAddVecMulVal<3>(const real* a, const real* b, const real s, real* c) {
	c[0] = a[0] + b[0] * s;
	c[1] = a[1] + b[1] * s;
	c[2] = a[2] + b[2] * s;
}

// a = (b + a * s0) * s1 -- Kui: this is customized function call
template <int dim>
__forceinline__ __device__ void VecAddVecMulValMulVal(const real* b, const real s0, const real s1, real* a);

template <>
__forceinline__ __device__ void VecAddVecMulValMulVal<2>(const real* b, const real s0, const real s1, real* a) {
	a[0] = (b[0] + a[0] * s0) * s1;
	a[1] = (b[1] + a[1] * s0) * s1;
}

template <>
__forceinline__ __device__ void VecAddVecMulValMulVal<3>(const real* b, const real s0, const real s1, real* a) {
	a[0] = (b[0] + a[0] * s0) * s1;
	a[1] = (b[1] + a[1] * s0) * s1;
	a[2] = (b[2] + a[2] * s0) * s1;
}

// c = a * s0 + b * s1
template <int dim>
__forceinline__ __device__ void VecMulValAddVecMulVal(const real* a, const real s0, const real* b, const real s1, real* c);

template <>
__forceinline__ __device__ void VecMulValAddVecMulVal<2>(const real* a, const real s0, const real* b, const real s1, real* c) {
	c[0] = a[0] * s0 + b[0] * s1;
	c[1] = a[1] * s0 + b[1] * s1;
}

template <>
__forceinline__ __device__ void VecMulValAddVecMulVal<3>(const real* a, const real s0, const real* b, const real s1, real* c) {
	c[0] = a[0] * s0 + b[0] * s1;
	c[1] = a[1] * s0 + b[1] * s1;
	c[2] = a[2] * s0 + b[2] * s1;
}

// norm, which returns the square root of squaredNorm 
template <int dim>
__forceinline__ __device__ real VecNorm(const real* a);

template <>
__forceinline__ __device__ real VecNorm<2>(const real* a) {
	return sqrtf(a[0] * a[0] + a[1] * a[1]);
}

template <>
__forceinline__ __device__ real VecNorm<3>(const real* a) {
	return sqrtf(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

// squaredNorm
template <int dim>
__forceinline__ __device__ real VecSquaredNorm(const real* a);

template <>
__forceinline__ __device__ real VecSquaredNorm<2>(const real* a) {
	return (a[0] * a[0] + a[1] * a[1]);
}

template <>
__forceinline__ __device__ real VecSquaredNorm<3>(const real* a) {
	return (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

// vector dot vector
template <int dim>
__forceinline__ __device__ real VecDot(const real* a, const real* b);

template <>
__forceinline__ __device__ real VecDot<2>(const real* a, const real* b) {
	return (a[0] * b[0] + a[1] * b[1]);
}

template <>
__forceinline__ __device__ real VecDot<3>(const real* a, const real* b) {
	return (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
}

// b = a 
template <int dim>
__forceinline__ __device__ void VecAssign(const real* a, real* b);

template <>
__forceinline__ __device__ void VecAssign<2>(const real* a, real* b) {
	b[0] = a[0]; 
	b[1] = a[1];
}

template <>
__forceinline__ __device__ void VecAssign<3>(const real* a, real* b) {
	b[0] = a[0]; 
	b[1] = a[1];
	b[2] = a[2];
}

// C = a * b^T
template <int dim>
__forceinline__ __device__ void VecMulVecT(const real* a, const real* b, real* C);

template <>
__forceinline__ __device__ void VecMulVecT<2>(const real* a, const real* b, real* C) {
    C[0] = a[0] * b[0]; C[2] = a[0] * b[1];
    C[1] = a[1] * b[0]; C[3] = a[1] * b[1];
}

template <>
__forceinline__ __device__ void VecMulVecT<3>(const real* a, const real* b, real* C) {
	C[0] = a[0] * b[0]; C[3] = a[0] * b[1]; C[6] = a[0] * b[2];
	C[1] = a[1] * b[0]; C[4] = a[1] * b[1]; C[7] = a[1] * b[2];
	C[2] = a[2] * b[0]; C[5] = a[2] * b[1]; C[8] = a[2] * b[2];
}

// c = a^T * B
template <int dim>
__forceinline__ __device__ void VecTMulMat(const real* a, const real* B, real* c);

template <>
__forceinline__ __device__ void VecTMulMat<2>(const real* a, const real* B, real* c) {
    c[0] = a[0] * B[0] + a[1] * B[1];
    c[1] = a[0] * B[2] + a[1] * B[3];
}

template <>
__forceinline__ __device__ void VecTMulMat<3>(const real* a, const real* B, real* c) {
	c[0] = a[0] * B[0] + a[1] * B[1] + a[2] * B[2];
	c[1] = a[0] * B[3] + a[1] * B[4] + a[2] * B[5];
	c[2] = a[0] * B[6] + a[1] * B[7] + a[2] * B[8];
}

// C = A * s
template <int dim>
__forceinline__ __device__ void MatMulVal(const real* A, const real s, real* C);

template <>
__forceinline__ __device__ void MatMulVal<2>(const real* A, const real s, real* C) {
    C[0] = A[0] * s; C[2] = A[2] * s;
    C[1] = A[1] * s; C[3] = A[3] * s;
}

template <>
__forceinline__ __device__ void MatMulVal<3>(const real* A, const real s, real* C) {
	C[0] = A[0] * s; C[3] = A[3] * s; C[6] = A[6] * s;
	C[1] = A[1] * s; C[4] = A[4] * s; C[7] = A[7] * s;
	C[2] = A[2] * s; C[5] = A[5] * s; C[8] = A[8] * s;
}

// C += s * a * b^T
template <int dim>
__forceinline__ __device__ void MatAddonVecVecTVal(const real* a, const real* b, const real s, real* C);

template <>
__forceinline__ __device__ void MatAddonVecVecTVal<2>(const real* a, const real* b, const real s, real* C) {
	C[0] += s * a[0] * b[0]; C[2] += s * a[0] * b[1];
	C[1] += s * a[1] * b[0]; C[3] += s * a[1] * b[1];
}

template <>
__forceinline__ __device__ void MatAddonVecVecTVal<3>(const real* a, const real* b, const real s, real* C) {
	C[0] += s * a[0] * b[0]; C[3] += s * a[0] * b[1]; C[6] += s * a[0] * b[2];
	C[1] += s * a[1] * b[0]; C[4] += s * a[1] * b[1]; C[7] += s * a[1] * b[2];
	C[2] += s * a[2] * b[0]; C[5] += s * a[2] * b[1]; C[8] += s * a[2] * b[2];
}

// B = A^T
template <int dim>
__forceinline__ __device__ void MatT(const real* A, real* B);

template <>
__forceinline__ __device__ void MatT<2>(const real* A, real* B) {
	B[0] = A[0]; B[2] = A[1];
	B[1] = A[2]; B[3] = A[3];
}

template <>
__forceinline__ __device__ void MatT<3>(const real* A, real* B) {
	B[0] = A[0]; B[3] = A[1]; B[6] = A[2];
	B[1] = A[3]; B[4] = A[4]; B[7] = A[5];
	B[2] = A[6]; B[5] = A[7]; B[8] = A[8];
}

// trace
template <int dim>
__forceinline__ __device__ real MatTrace(const real* C);

template <>
__forceinline__ __device__ real MatTrace<2>(const real* C) {
	return (C[0] + C[3]);
}

template <>
__forceinline__ __device__ real MatTrace<3>(const real* C) {
	return (C[0] + C[4] + C[8]);
}

// C = A
template <int dim>
__forceinline__ __device__ void MatAssign(const real* A, real* C);

template <>
__forceinline__ __device__ void MatAssign<2>(const real* A, real* C) {
	C[0] = A[0]; C[2] = A[2];
	C[1] = A[1]; C[3] = A[3];
}

template <>
__forceinline__ __device__ void MatAssign<3>(const real* A, real* C) {
	C[0] = A[0]; C[3] = A[3]; C[6] = A[6];
	C[1] = A[1]; C[4] = A[4]; C[7] = A[7];
	C[2] = A[2]; C[5] = A[5]; C[8] = A[8];
}

// c = A * b
template <int dim>
__forceinline__ __device__ void MatMulVec(const real* A, const real* b, real* c);

template <>
__forceinline__ __device__ void MatMulVec<2>(const real* A, const real* b, real* c) {
    c[0] = A[0] * b[0] + A[2] * b[1];
    c[1] = A[1] * b[0] + A[3] * b[1];
}

template <>
__forceinline__ __device__ void MatMulVec<3>(const real* A, const real* b, real* c) {
	c[0] = A[0] * b[0] + A[3] * b[1] + A[6] * b[2];
	c[1] = A[1] * b[0] + A[4] * b[1] + A[7] * b[2];
	c[2] = A[2] * b[0] + A[5] * b[1] + A[8] * b[2];
}

// c = A^T * b
template <int dim>
__forceinline__ __device__ void MatTMulVec(const real* A, const real* b, real* c);

template <>
__forceinline__ __device__ void MatTMulVec<2>(const real* A, const real* b, real* c) {
    c[0] = A[0] * b[0] + A[1] * b[1];
    c[1] = A[2] * b[0] + A[3] * b[1];
}

template <>
__forceinline__ __device__ void MatTMulVec<3>(const real* A, const real* b, real* c) {
	c[0] = A[0] * b[0] + A[1] * b[1] + A[2] * b[2];
	c[1] = A[3] * b[0] + A[4] * b[1] + A[5] * b[2];
	c[2] = A[6] * b[0] + A[7] * b[1] + A[8] * b[2];
}

// C = A * B
template <int dim>
__forceinline__ __device__ void MatMulMat(const real* A, const real* B, real* C);

template <>
__forceinline__ __device__ void MatMulMat<2>(const real* A, const real* B, real* C) {
    C[0] = A[0] * B[0] + A[2] * B[1]; C[2] = A[0] * B[2] + A[2] * B[3];
    C[1] = A[1] * B[0] + A[3] * B[1]; C[3] = A[1] * B[2] + A[3] * B[3];
}

template <>
__forceinline__ __device__ void MatMulMat<3>(const real* A, const real* B, real* C) {
	C[0] = A[0] * B[0] + A[3] * B[1] + A[6] * B[2]; C[3] = A[0] * B[3] + A[3] * B[4] + A[6] * B[5]; C[6] = A[0] * B[6] + A[3] * B[7] + A[6] * B[8];
	C[1] = A[1] * B[0] + A[4] * B[1] + A[7] * B[2]; C[4] = A[1] * B[3] + A[4] * B[4] + A[7] * B[5]; C[7] = A[1] * B[6] + A[4] * B[7] + A[7] * B[8];
	C[2] = A[2] * B[0] + A[5] * B[1] + A[8] * B[2]; C[5] = A[2] * B[3] + A[5] * B[4] + A[8] * B[5]; C[8] = A[2] * B[6] + A[5] * B[7] + A[8] * B[8];
}

// C = A * B^T
template <int dim>
__forceinline__ __device__ void MatMulMatT(const real* A, const real* B, real* C);

template <>
__forceinline__ __device__ void MatMulMatT<2>(const real* A, const real* B, real* C) {
    C[0] = A[0] * B[0] + A[2] * B[2]; C[2] = A[0] * B[1] + A[2] * B[3];
    C[1] = A[1] * B[0] + A[3] * B[2]; C[3] = A[1] * B[1] + A[3] * B[3];
}

template <>
__forceinline__ __device__ void MatMulMatT<3>(const real* A, const real* B, real* C) {
	C[0] = A[0] * B[0] + A[3] * B[3] + A[6] * B[6]; C[3] = A[0] * B[1] + A[3] * B[4] + A[6] * B[7]; C[6] = A[0] * B[2] + A[3] * B[5] + A[6] * B[8];
	C[1] = A[1] * B[0] + A[4] * B[3] + A[7] * B[6]; C[4] = A[1] * B[1] + A[4] * B[4] + A[7] * B[7]; C[7] = A[1] * B[2] + A[4] * B[5] + A[7] * B[8];
	C[2] = A[2] * B[0] + A[5] * B[3] + A[8] * B[6]; C[5] = A[2] * B[1] + A[5] * B[4] + A[8] * B[7]; C[8] = A[2] * B[2] + A[5] * B[5] + A[8] * B[8];
}

// C = A^T * B
template <int dim>
__forceinline__ __device__ void MatTMulMat(const real* A, const real* B, real* C);

template <>
__forceinline__ __device__ void MatTMulMat<2>(const real* A, const real* B, real* C) {
	C[0] = A[0] * B[0] + A[1] * B[1]; C[2] = A[0] * B[2] + A[1] * B[3];
	C[1] = A[2] * B[0] + A[3] * B[1]; C[3] = A[2] * B[2] + A[3] * B[3];
}

template <>
__forceinline__ __device__ void MatTMulMat<3>(const real* A, const real* B, real* C) {
	C[0] = A[0] * B[0] + A[1] * B[1] + A[2] * B[2]; C[3] = A[0] * B[3] + A[1] * B[4] + A[2] * B[5]; C[6] = A[0] * B[6] + A[1] * B[7] + A[2] * B[8];
	C[1] = A[3] * B[0] + A[4] * B[1] + A[5] * B[2]; C[4] = A[3] * B[3] + A[4] * B[4] + A[5] * B[5]; C[7] = A[3] * B[6] + A[4] * B[7] + A[5] * B[8];
	C[2] = A[6] * B[0] + A[7] * B[1] + A[8] * B[2]; C[5] = A[6] * B[3] + A[7] * B[4] + A[8] * B[5]; C[8] = A[6] * B[6] + A[7] * B[7] + A[8] * B[8];
}

// C = A - B
template <int dim>
__forceinline__ __device__ void MatSubMat(const real* A, const real* B, real* C);

template <>
__forceinline__ __device__ void MatSubMat<2>(const real* A, const real* B, real* C) {
    C[0] = A[0] - B[0]; C[2] = A[2] - B[2];
    C[1] = A[1] - B[1]; C[3] = A[3] - B[3];
}

template <>
__forceinline__ __device__ void MatSubMat<3>(const real* A, const real* B, real* C) {
	C[0] = A[0] - B[0]; C[3] = A[3] - B[3]; C[6] = A[6] - B[6];
	C[1] = A[1] - B[1]; C[4] = A[4] - B[4]; C[7] = A[7] - B[7];
	C[2] = A[2] - B[2]; C[5] = A[5] - B[5]; C[8] = A[8] - B[8];
}


// C += B
template <int dim>
__forceinline__ __device__ void MatAddonMat(const real* B, real* C);

template <>
__forceinline__ __device__ void MatAddonMat<2>(const real* B, real* C) {
    C[0] += B[0]; C[2] += B[2];
    C[1] += B[1]; C[3] += B[3];
}

template <>
__forceinline__ __device__ void MatAddonMat<3>(const real* B, real* C) {
	C[0] += B[0]; C[3] += B[3];	C[6] += B[6];
	C[1] += B[1]; C[4] += B[4];	C[7] += B[7];
	C[2] += B[2]; C[5] += B[5];	C[8] += B[8];
}

// B += I * s
template <int dim>
__forceinline__ __device__ void MatAddonIdentity(const real s, real* B);

template <>
__forceinline__ __device__ void MatAddonIdentity<2>(const real s, real* B) {
	B[0] += s; 
	B[3] += s;
}

template <>
__forceinline__ __device__ void MatAddonIdentity<3>(const real s, real* B) {
	B[0] += s;
	B[4] += s;
	B[8] += s;
}

// C = A + B * s
template <int dim>
__forceinline__ __device__ void MatAddMatMulVal(const real* A, const real* B, const real s, real* C);

template <>
__forceinline__ __device__ void MatAddMatMulVal<2>(const real* A, const real* B, const real s, real* C) {
    C[0] = A[0] + B[0] * s; C[2] = A[2] + B[2] * s;
    C[1] = A[1] + B[1] * s; C[3] = A[3] + B[3] * s;
}

template <>
__forceinline__ __device__ void MatAddMatMulVal<3>(const real* A, const real* B, const real s, real* C) {
	C[0] = A[0] + B[0] * s; C[3] = A[3] + B[3] * s; C[6] = A[6] + B[6] * s;
	C[1] = A[1] + B[1] * s; C[4] = A[4] + B[4] * s; C[7] = A[7] + B[7] * s;
	C[2] = A[2] + B[2] * s; C[5] = A[5] + B[5] * s; C[8] = A[8] + B[8] * s;
}

// C = (1 - a) * A + a * B
template <int dim>
__forceinline__ __device__ void MatLERPMat(const real* A, const real* B, const real alpha, real* C);

template <>
__forceinline__ __device__ void MatLERPMat<2>(const real* A, const real* B, const real alpha, real* C) {
    C[0] = (1.0f - alpha) * A[0] + alpha * B[0]; C[2] = (1.0f - alpha) * A[2] + alpha * B[2];
    C[1] = (1.0f - alpha) * A[1] + alpha * B[1]; C[3] = (1.0f - alpha) * A[3] + alpha * B[3];
}

template <>
__forceinline__ __device__ void MatLERPMat<3>(const real* A, const real* B, const real alpha, real* C) {
	C[0] = (1.0f - alpha) * A[0] + alpha * B[0]; C[3] = (1.0f - alpha) * A[3] + alpha * B[3]; C[6] = (1.0f - alpha) * A[6] + alpha * B[6];
	C[1] = (1.0f - alpha) * A[1] + alpha * B[1]; C[4] = (1.0f - alpha) * A[4] + alpha * B[4]; C[7] = (1.0f - alpha) * A[7] + alpha * B[7];
	C[2] = (1.0f - alpha) * A[2] + alpha * B[2]; C[5] = (1.0f - alpha) * A[5] + alpha * B[5]; C[8] = (1.0f - alpha) * A[8] + alpha * B[8];
}

// Det(A)
template <int dim>
__forceinline__ __device__ real MatDet(const real* A);

template <>
__forceinline__ __device__ real MatDet<2>(const real* A) {
    return (A[0] * A[3] - A[1] * A[2]);
}

template <>
__forceinline__ __device__ real MatDet<3>(const real* A) {
	real a00 = A[4] * A[8] - A[7] * A[5];			 		
	real a10 = A[7] * A[2] - A[1] * A[8];			  			
	real a20 = A[1] * A[5] - A[4] * A[2];

	return (a00 * A[0] + a10 * A[3] + a20 * A[6]);
}

// inv(A)
template <int dim>
__forceinline__ __device__ void MatInv(const real* A, real* A_inv);

template <>
__forceinline__ __device__ void MatInv<2>(const real* A, real* A_inv) {
    real inv_det = 1.0f / MatDet<2>(A);

    A_inv[0] = inv_det * A[3];
    A_inv[1] = -inv_det * A[1];
    A_inv[2] = -inv_det * A[2];
    A_inv[3] = inv_det * A[0];
}

template <>
__forceinline__ __device__ void MatInv<3>(const real* A, real* A_inv) {
	real a00 = A[4] * A[8] - A[7] * A[5];
	real a01 = A[6] * A[5] - A[3] * A[8];
	real a02 = A[3] * A[7] - A[6] * A[4];

	real a10 = A[7] * A[2] - A[1] * A[8];
	real a11 = A[0] * A[8] - A[6] * A[2];
	real a12 = A[6] * A[1] - A[0] * A[7];

	real a20 = A[1] * A[5] - A[4] * A[2];
	real a21 = A[2] * A[3] - A[0] * A[5];
	real a22 = A[0] * A[4] - A[3] * A[1];

	const real det = (a00 * A[0] + a10 * A[3] + a20 * A[6]);

	const real inv_det = 1.0f / det;

	A_inv[0] = a00 * inv_det; A_inv[3] = a01 * inv_det; A_inv[6] = a02 * inv_det;
	A_inv[1] = a10 * inv_det; A_inv[4] = a11 * inv_det; A_inv[7] = a12 * inv_det;
	A_inv[2] = a20 * inv_det; A_inv[5] = a21 * inv_det; A_inv[8] = a22 * inv_det;
}

// CwiseProduct
template <int dim>
__forceinline__ __device__ void MatMatCwiseProduct(const real* A, const real* B, real* C);

template <>
__forceinline__ __device__ void MatMatCwiseProduct<2>(const real* A, const real* B, real* C)
{
    C[0] = A[0] * B[0];
    C[1] = A[1] * B[1];
    C[2] = A[2] * B[2];
    C[3] = A[3] * B[3];
}

template <>
__forceinline__ __device__ void MatMatCwiseProduct<3>(const real* A, const real* B, real* C)
{
	C[0] = A[0] * B[0];
	C[1] = A[1] * B[1];
	C[2] = A[2] * B[2];
	C[3] = A[3] * B[3];
	C[4] = A[4] * B[4];
	C[5] = A[5] * B[5];
	C[6] = A[6] * B[6];
	C[7] = A[7] * B[7];
	C[8] = A[8] * B[8];
}

// CwiseProductSum
template <int dim>
__forceinline__ __device__ real MatMatCwiseProductSum(const real* A, const real* B);

template <>
__forceinline__ __device__ real MatMatCwiseProductSum<2>(const real* A, const real* B) {
    return (A[0] * B[0] + A[1] * B[1] + A[2] * B[2] + A[3] * B[3]);
}

template <>
__forceinline__ __device__ real MatMatCwiseProductSum<3>(const real* A, const real* B) {
	return (A[0] * B[0] + A[1] * B[1] + A[2] * B[2] + A[3] * B[3] + A[4] * B[4] + A[5] * B[5] + A[6] * B[6] + A[7] * B[7] + A[8] * B[8]);
}

// dRFromdF
template <int dim>
__forceinline__ __device__ void dRFromdF(const real* F, const real* R, const real* S, const real* dF, real* dR);

template <>
__forceinline__ __device__ void dRFromdF<2>(const real* F, const real* R, const real* S, const real* dF, real* dR) {
    // set W = R^T dR = [  0    x  ]
    //                  [  -x   0  ]
    //
    // R^T dF - dF^T R = WS + SW
    //
    // WS + SW = [ x(s21 - s12)   x(s11 + s22) ]
    //           [ -x[s11 + s22]  x(s21 - s12) ]
    // ----------------------------------------------------
    real lhs[4];
    MatTMulMat<2>(R, dF, lhs);

    real tmpMat[4];
    MatTMulMat<2>(dF, R, tmpMat);
    MatSubMat<2>(lhs, tmpMat, lhs);

    const real x = lhs[2] / (S[0] + S[3]);
    real W[4] = {0, -x, x, 0};

    MatMulMat<2>(R, W, dR);
}

template <>
__forceinline__ __device__ void dRFromdF<3>(const real* F, const real* R, const real* S, const real* dF, real* dR) {
	real lhs[9], tmpMat[9];

	// lhs = R^T * dF - dF^T * r;
	MatTMulMat<3>(R, dF, tmpMat);
	MatTMulMat<3>(dF, R, lhs);
	MatSubMat<3>(tmpMat, lhs, lhs);

	// https://www.overleaf.com/read/rxssbpcxjypz.
	real A[9] = { 0 };
	A[0] = S[0] + S[4];
	A[4] = S[0] + S[8];
	A[8] = S[4] + S[8];
	A[3] = A[1] = S[7];
	A[6] = A[2] =-S[6];
	A[7] = A[5] = S[3];

	real A_inv[9]; MatInv<3>(A, A_inv); 
	const real b[3] = { lhs[0 + 3 * 1], lhs[0 + 3 * 2], lhs[1 + 3 * 2] };
	real xyz[3]; MatMulVec<3>(A_inv, b, xyz);

	real W[9] = {0};
	W[0] = W[4] = W[8] = 0;
	W[3] =  xyz[0]; W[6] =  xyz[1];
	W[1] = -xyz[0]; W[7] =  xyz[2];
	W[2] = -xyz[1]; W[5] = -xyz[2];

	MatMulMat<3>(R, W, dR);
}

#endif
