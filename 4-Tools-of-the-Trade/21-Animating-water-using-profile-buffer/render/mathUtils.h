
#ifndef _MATH_UTILS_
#define _MATH_UTILS_

#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>

#include "cyPoint.h"
#include "cyMatrix3.h"
#include "cyMatrix4.h"

#pragma warning( disable : 26812 )

#define for0(VAR, MAX) for (std::remove_const<decltype(MAX)>::type VAR = 0; VAR < (MAX); VAR++)

using Mat3 = cyMatrix3f;
using Mat4 = cyMatrix4f;

using Vec2 = cyPoint2f;
using Vec3 = cyPoint3f;
using Vec4 = cyPoint4f;

template<class T>
inline void get_barycentric(T x, int& i, T& f, int i_low, int i_high)
{
	double s = std::floor(x);
	i = (int)s;
	if (i < i_low) {
		i = i_low;
		f = 0;
	}
	else if (i > i_high - 2) {
		i = i_high - 2;
		f = 1;
	}
	else {
		f = (T)(x - s);
	}
}

template<class S, class T>
inline S lerp(const S& value0, const S& value1, T f)
{
	return (1 - f) * value0 + f * value1;
}

template<class S, class T>
inline S bilerp(const S& v00, const S& v10,
	const S& v01, const S& v11,
	T fx, T fy)
{
	return lerp(
		lerp(v00, v10, fx),
		lerp(v01, v11, fx),
		fy);
}

template<class S, class T>
inline S trilerp(
	const S& v000, const S& v100,
	const S& v010, const S& v110,
	const S& v001, const S& v101,
	const S& v011, const S& v111,
	T fx, T fy, T fz)
{
	return lerp(
		bilerp(v000, v100, v010, v110, fx, fy),
		bilerp(v001, v101, v011, v111, fx, fy),
		fz);
}

template<class S, class T>
inline S quadlerp(
	const S& v0000, const S& v1000,
	const S& v0100, const S& v1100,
	const S& v0010, const S& v1010,
	const S& v0110, const S& v1110,
	const S& v0001, const S& v1001,
	const S& v0101, const S& v1101,
	const S& v0011, const S& v1011,
	const S& v0111, const S& v1111,
	T fx, T fy, T fz, T ft)
{
	return lerp(
		trilerp(v0000, v1000, v0100, v1100, v0010, v1010, v0110, v1110, fx, fy, fz),
		trilerp(v0001, v1001, v0101, v1101, v0011, v1011, v0111, v1111, fx, fy, fz),
		ft);
}

template<class S, class T>
inline S cubicinterp(S qneg1, S q0, S q1, S q2, T x)
{
	//@@@ Implement the minmod-limited cubic Hermite interpolation in 1 dimension
	//@@@ The q data is assumed to be given at grid points -1, 0, 1, and 2, and
	//@@@ the parameter x is somewhere between 0 and 1. Evaluate the cubic at x and
	//@@@ return the value.
	S d0 = (q1 - qneg1) / 2;
	S d1 = (q2 - q0) / 2;
	S d = q1 - q0;

	if ((d >= 0 && d1 < 0) || (d <= 0 && d1 > 0))
		d1 = 0;

	if ((d >= 0 && d0 < 0) || (d <= 0 && d0 > 0))
		d0 = 0;

	return (d0 + d1 - 2 * d) * x * x * x + (3 * d - 2 * d0 - d1) * x * x + d0 * x + q0;

}
#endif