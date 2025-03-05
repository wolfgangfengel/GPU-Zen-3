#pragma once

#include "../config_type.h"
#include "Grid.h"
#include "Defines.h"

#define ScaleDim 8     // 64x64 --8x8--> 8x8
#define ScaleSize  64  // 8x8
#define SSWE_SHARE_FUNC inline __device__ __host__
#define SSWE_HOST_ONLY_FUNC  inline __host__

namespace SSWEShareCode
{
	/* \brief cubic bump weight function
	*/
	SSWE_SHARE_FUNC real cubic_bump(real x)
	{
		if (abs(x) >= 1)
			return 0.0f;
		else
			return x * x * (2 * abs(x) - 3) + 1;
	}

	/* \brief Gerstner wave
	*/
	SSWE_SHARE_FUNC void gerstner_wave(real* gw, real phase /*=knum*x*/, real knum)
	{
		real s = sin(phase);
		real c = cos(phase);
		gw[0] = -s;
		gw[1] = c;
		gw[2] = -knum * c;
		gw[3] = -knum * s;
	}

	/* \brief The dispersion relation of deep water waves, see https://en.wikipedia.org/wiki/Dispersion_relation
	*/
	SSWE_SHARE_FUNC real dispersionRelation(real k, real s)
	{
		return s * sqrt(GRAVITY * k);
	}

	/* \brief The Phillips spectrum
	*/
	SSWE_SHARE_FUNC real phillips_spectrum(real k)
	{
		real m_windSpeed = 10.f;
		real A = TAU / k;
		real B = expf(-1.8038897788076411f * A * A / powf(m_windSpeed, 4));
		return 0.139098f * sqrtf(A * B);
	}

	/* \brief Interpolate two Gerstner waves
	*/
	SSWE_SHARE_FUNC void gerstner_wave_interpolated(real* gw, real k, real p, real L, real scale, real t)
	{
		real phase1 = k * p - dispersionRelation(k, scale) * t;
		real phase2 = k * (p - L) - dispersionRelation(k, scale) * t;

		real weight1 = p / L;
		real weight2 = 1 - weight1;

		real gw1[WAVE_DIM];
		real gw2[WAVE_DIM];

		gerstner_wave(gw1, phase1, k);
		gerstner_wave(gw2, phase2, k);

		real cb1 = cubic_bump(weight1);
		real cb2 = cubic_bump(weight2);

		for (int i = 0; i < WAVE_DIM; ++i)
		{
			gw[i] = cb1 * gw1[i] + cb2 * gw2[i];
		}
	}
	/* \brief Compute each entry of profile buffer
	\param result: current entry result
	\param k: current wave number
	\param p: current position (scaled to profile buffer space)
	\param L: profile buffer length
	\param scale: profile scale
	\param t: current time
	*/
	SSWE_SHARE_FUNC void PB_Entry(real* result, real k, real p, real L, real scale, real t)
	{
		real lambda = TAU / k;
		real gw[WAVE_DIM];
		real sp = phillips_spectrum(k);
		gerstner_wave_interpolated(gw, k, p, L, scale, t);

		for (int i = 0; i < WAVE_DIM; ++i)
		{
			result[i] += lambda * sp * gw[i];
		}
	}

	/* \brief Compute group velocity
	*/
	SSWE_SHARE_FUNC void GV(real zeta, real* val)
	{
		real lambda = powf(2, zeta);
		real k = TAU / lambda;
		real cg = .5f * sqrt(GRAVITY / k);
		real density = phillips_spectrum(k);
		val[0] = cg * density;
		val[1] = density;
	}

	SSWE_SHARE_FUNC void Precompute_Profile_Buffer(real* PB_field,
		real t, real k_min, real k_max, real scale, real L, int integration_nodes,
		Grid grid, const int idx1D)
	{
		real p = (idx1D * L) / PB_RESOLUTION;
		real dk = (k_max - k_min) / integration_nodes;
		real k = k_min + 0.5f * dk;
		real result[WAVE_DIM] = { 0.f };
		PB_Entry(result, k, p, L, scale, t);
		for (int i = 1; i < integration_nodes; ++i)
		{
			k += dk;
			PB_Entry(result, k, p, L, scale, t);
		}
		for (int i = 0; i < WAVE_DIM; ++i)
		{
			PB_field[idx1D + PB_RESOLUTION * i] = dk * result[i];
		}
	}

	/* \brief Compute group velocity
	*/
	SSWE_SHARE_FUNC void Precompute_Group_Speeds(real* gv,
		int integration_nodes, real k_min, real k_max, Grid grid)
	{
		real zeta_min = log2f(TAU / k_max);
		real zeta_max = log2f(TAU / k_min);
		real dzeta = (zeta_max - zeta_min) / integration_nodes;
		real zeta = zeta_min + 0.5f * dzeta;
		real val[2] = { 0.f, 0.f };
		GV(zeta, val);
		gv[0] = val[0];
		gv[1] = val[1];
		for (int i = 1; i < integration_nodes; ++i)
		{
			zeta += dzeta;
			GV(zeta, val);
			gv[0] += val[0];
			gv[1] += val[1];
		}
		gv[0] *= dzeta;
		gv[1] *= dzeta;
	}
}  // SSWEShareCode End