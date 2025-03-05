__device__ __forceinline__ void CU_Swap2x2(float& a, float& b)
{
	float tmp = a;
	a = b;
	b = tmp;
}

__device__ __forceinline__
void svd2(const float* A, float* U, float* sigma, float* V)
{
	float gUc, gUs, gVc, gVs;
	float S_Sym[4];
	float xx[2]; xx[0] = __fadd_rn(A[0], A[3]); xx[1] = __fsub_rn(A[1], A[2]);

	float denominator = __fsqrt_rn(__fadd_rn(xx[0] * xx[0], xx[1] * xx[1]));

	gUc = 1;  gUs = 0;
	if (denominator != 0) { gUc = __fdiv_rn(xx[0], denominator); gUs = __fdiv_rn(-xx[1], denominator); }

	S_Sym[0] = A[0]; S_Sym[1] = A[1]; S_Sym[2] = A[2]; S_Sym[3] = A[3];

	float tau1 = S_Sym[0];
	float tau2 = S_Sym[1];
	S_Sym[0] = __fsub_rn(gUc * tau1, gUs * tau2);
	S_Sym[1] = __fadd_rn(gUs * tau1, gUc * tau2);

	tau1 = S_Sym[2];
	tau2 = S_Sym[3];
	S_Sym[2] = __fsub_rn(gUc * tau1, gUs * tau2);
	S_Sym[3] = __fadd_rn(gUs * tau1, gUc * tau2);

	float cosine, sine;
	float x = S_Sym[0], y = S_Sym[1], z = S_Sym[3];
	if (fabsf(y) < 1e-8f) {
		cosine = 1.0f;
		sine = 0;
		sigma[0] = x;
		sigma[1] = z;
	}
	else
	{
		float tau = 0.5f * __fsub_rn(x, z);
		float w = __fsqrt_rn(__fadd_rn(tau * tau, y * y));
		float t;
		if (tau > 0)    t = __fdiv_rn(y, __fadd_rn(tau, w));
		else            t = __fdiv_rn(y, __fsub_rn(tau, w));

		cosine = __frsqrt_rn(__fadd_rn(t * t, 1));
		sine = -t * cosine;

		float c2 = cosine * cosine;
		float csy = 2.0f * cosine * sine * y;
		float s2 = sine * sine;
		sigma[0] = c2 * x - csy + s2 * z;
		sigma[1] = s2 * x + csy + c2 * z;
	}

	if (sigma[0] < sigma[1])
	{
		CU_Swap2x2(sigma[0], sigma[1]);
		gVc = -sine; gVs = cosine;
	}
	else
	{
		gVc = cosine; gVs = sine;
	}

	float new_c = __fsub_rn(gUc * gVc, gUs * gVs), new_s = __fadd_rn(gUs * gVc, gUc * gVs);
	gUc = new_c;    gUs = new_s;

	U[0] = gUc;        U[2] = gUs;
	U[1] = -gUs;    U[3] = gUc;

	V[0] = gVc;        V[2] = gVs;
	V[1] = -gVs;    V[3] = gVc;
}
