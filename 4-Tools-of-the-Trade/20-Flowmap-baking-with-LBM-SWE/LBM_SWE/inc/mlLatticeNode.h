#pragma once

#ifndef _MLLATTICENODE_
#define _MLLATTICENODE_
#include "mlCoreWin.h"
template<class T, int count> //count specify the number of distribution functions for each node
class mlLatticeNode
{
public:
	T f[count];//distribution functions in any dimentions

	MLFUNC_TYPE   T& operator () (int i);
	MLFUNC_TYPE   const T& operator () (int i)const;
	MLFUNC_TYPE   T& operator [] (int i);
	MLFUNC_TYPE   const T& operator [] (int i)const;

	MLFUNC_TYPE   mlLatticeNode<T, count>& operator = (const mlLatticeNode<T, count>& node);
	MLFUNC_TYPE   mlLatticeNode<T, count>& operator = (const T& value);

	MLFUNC_TYPE   mlLatticeNode<T, count>& operator += (const mlLatticeNode<T, count>& node);
	MLFUNC_TYPE   mlLatticeNode<T, count>   operator +  (const mlLatticeNode<T, count>& node);
	MLFUNC_TYPE   mlLatticeNode<T, count>& operator += (const T& value);
	MLFUNC_TYPE   mlLatticeNode<T, count>   operator +  (const T& value);

	MLFUNC_TYPE   mlLatticeNode<T, count>& operator -= (const mlLatticeNode<T, count>& node);
	MLFUNC_TYPE   mlLatticeNode<T, count>   operator -  (const mlLatticeNode<T, count>& node);
	MLFUNC_TYPE   mlLatticeNode<T, count>& operator -= (const T& value);
	MLFUNC_TYPE   mlLatticeNode<T, count>   operator -  (const T& value);

	MLFUNC_TYPE   mlLatticeNode<T, count>& operator *= (const mlLatticeNode<T, count>& node);
	MLFUNC_TYPE   mlLatticeNode<T, count>   operator *  (const mlLatticeNode<T, count>& node);
	MLFUNC_TYPE   mlLatticeNode<T, count>& operator *= (const T& value);
	MLFUNC_TYPE   mlLatticeNode<T, count>   operator *  (const T& value);

	MLFUNC_TYPE   mlLatticeNode<T, count>& operator /= (const mlLatticeNode<T, count>& node);
	MLFUNC_TYPE   mlLatticeNode<T, count>   operator /  (const mlLatticeNode<T, count>& node);
	MLFUNC_TYPE   mlLatticeNode<T, count>& operator /= (const T& value);
	MLFUNC_TYPE   mlLatticeNode<T, count>   operator /  (const T& value);

	MLFUNC_TYPE   mlLatticeNode();
	MLFUNC_TYPE   mlLatticeNode(const mlLatticeNode& node);
};

typedef mlLatticeNode<REAL, 5> mlLatticeNodeD2Q5f;
typedef mlLatticeNode<double, 5> mlLatticeNodeD2Q5d;

typedef mlLatticeNode<REAL, 7> mlLatticeNodeD2Q7f;
typedef mlLatticeNode<double, 7> mlLatticeNodeD2Q7d;

typedef mlLatticeNode<REAL, 9> mlLatticeNodeD2Q9f;
typedef mlLatticeNode<double, 9> mlLatticeNodeD2Q9d;

 

#define MlLatticeNodeD2Q5 mlLatticeNode<T,5>
#define MlLatticeNodeD2Q7 mlLatticeNode<T,7>
#define MlLatticeNodeD2Q9 mlLatticeNode<T,9>
 

///////////////////implementation//////////////////

template<class T, int count>
MLFUNC_TYPE  mlLatticeNode<T, count>::mlLatticeNode()
{
	/*memset(f, 0, sizeof(T)*count);*/
	for (int i = 0; i < count; i++)
		f[i] = 0;
}

template<class T, int count>
MLFUNC_TYPE  mlLatticeNode<T, count>::mlLatticeNode(const mlLatticeNode<T, count>& node)
{
	//memcpy(f, node.f, sizeof(T)*count);
	for (int i = 0; i < count; i++)
		f[i] = node.f[i];
}

template<class T, int count>
MLFUNC_TYPE  mlLatticeNode<T, count>& mlLatticeNode<T, count>::operator = (const mlLatticeNode<T, count>& node)
{
	//memcpy(f, node.f, sizeof(T)*count);
	for (int i = 0; i < count; i++)
		f[i] = node.f[i];

	return (*this);
}
template<class T, int count>
MLFUNC_TYPE  mlLatticeNode<T, count>& mlLatticeNode<T, count>::operator = (const T& value)
{
	for (int i = 0; i < count; i++)
		f[i] = value;
	return (*this);
}

template<class T, int count>
MLFUNC_TYPE T& mlLatticeNode<T, count>::operator () (int i)
{
	return f[i];
}

template<class T, int count>
MLFUNC_TYPE const T& mlLatticeNode<T, count>::operator () (int i)const
{
	return f[i];
}

template<class T, int count>
MLFUNC_TYPE T& mlLatticeNode<T, count>::operator [] (int i)
{
	return f[i];
}

template<class T, int count>
MLFUNC_TYPE const T& mlLatticeNode<T, count>::operator [] (int i)const
{
	return f[i];
}

template<class T, int count>
MLFUNC_TYPE mlLatticeNode<T, count>& mlLatticeNode<T, count>::operator += (const mlLatticeNode<T, count>& node)
{
	for (int i = 0; i < count; i++)
		f[i] += node.f[i];
	return (*this);
}

template<class T, int count>
MLFUNC_TYPE mlLatticeNode<T, count>   mlLatticeNode<T, count>::operator +  (const mlLatticeNode<T, count>& node)
{
	mlLatticeNode<T, count> temp(*this);
	return temp += node;
}

template<class T, int count>
MLFUNC_TYPE mlLatticeNode<T, count>& mlLatticeNode<T, count>::operator += (const T& value)
{
	for (int i = 0; i < count; i++)
		f[i] += value;
	return (*this);
}

template<class T, int count>
MLFUNC_TYPE mlLatticeNode<T, count>   mlLatticeNode<T, count>::operator +  (const T& value)
{
	mlLatticeNode<T, count> temp(*this);
	return temp += value;
}

template<class T, int count>
MLFUNC_TYPE mlLatticeNode<T, count>& mlLatticeNode<T, count>::operator -= (const mlLatticeNode<T, count>& node)
{
	for (int i = 0; i < count; i++)
		f[i] -= node.f[i];
	return (*this);
}

template<class T, int count>
MLFUNC_TYPE mlLatticeNode<T, count>   mlLatticeNode<T, count>::operator -  (const mlLatticeNode<T, count>& node)
{
	mlLatticeNode<T, count> temp(*this);
	return temp -= node;
}

template<class T, int count>
MLFUNC_TYPE mlLatticeNode<T, count>& mlLatticeNode<T, count>::operator -= (const T& value)
{
	for (int i = 0; i < count; i++)
		f[i] -= value;
	return (*this);
}

template<class T, int count>
MLFUNC_TYPE mlLatticeNode<T, count>   mlLatticeNode<T, count>::operator -  (const T& value)
{
	mlLatticeNode<T, count> temp(*this);
	return temp -= value;
}

template<class T, int count>
MLFUNC_TYPE mlLatticeNode<T, count>& mlLatticeNode<T, count>::operator *= (const mlLatticeNode<T, count>& node)
{
	for (int i = 0; i < count; i++)
		f[i] *= node.f[i];
	return (*this);
}

template<class T, int count>
MLFUNC_TYPE mlLatticeNode<T, count>   mlLatticeNode<T, count>::operator *  (const mlLatticeNode<T, count>& node)
{
	mlLatticeNode<T, count> temp(*this);
	return temp *= node;
}

template<class T, int count>
MLFUNC_TYPE mlLatticeNode<T, count>& mlLatticeNode<T, count>::operator *= (const T& value)
{
	for (int i = 0; i < count; i++)
		f[i] *= value;
	return (*this);
}

template<class T, int count>
MLFUNC_TYPE mlLatticeNode<T, count>   mlLatticeNode<T, count>::operator *  (const T& value)
{
	mlLatticeNode<T, count> temp(*this);
	return temp *= value;
}

template<class T, int count>
MLFUNC_TYPE mlLatticeNode<T, count>& mlLatticeNode<T, count>::operator /= (const mlLatticeNode<T, count>& node)
{
	for (int i = 0; i < count; i++)
		f[i] /= node.f[i];
	return (*this);
}

template<class T, int count>
MLFUNC_TYPE mlLatticeNode<T, count>   mlLatticeNode<T, count>::operator /  (const mlLatticeNode<T, count>& node)
{
	mlLatticeNode<T, count> temp(*this);
	return temp /= node;
}

template<class T, int count>
MLFUNC_TYPE mlLatticeNode<T, count>& mlLatticeNode<T, count>::operator /= (const T& value)
{
	for (int i = 0; i < count; i++)
		f[i] /= value;
	return (*this);
}

template<class T, int count>
MLFUNC_TYPE mlLatticeNode<T, count>   mlLatticeNode<T, count>::operator /  (const T& value)
{
	mlLatticeNode<T, count> temp(*this);
	return temp /= value;
}



#endif //_MLLATTICENODE_
