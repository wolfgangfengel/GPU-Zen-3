#pragma once
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <string>
#include <typeinfo>
#include <type_traits>
#include <omp.h>
//#include <objbase.h>
static const double INFTY = 1.0e32;

#define SMALLNUMBER	1.0e-5
#define HUGENUMBER	1.0e10
#define Sqr(x)		((x) * (x))

#define NumOfNeig 8 //map neighbor
#define NumOfbasis 5
#define Mach 340
#define Q 9
#define CS 5e-3  //Smagorinsky constants
#define Epsilon 20
#define M_PI 3.1415926*1
#define MLPI 3.1415926
#define  boundaryNum 4
 
//#pragma comment(lib,"3rdparty/opengl/glfw-3.2.1.bin.WIN64/lib-vc2015/glfw3.lib")
//#pragma comment(lib,"3rdparty/opengl/glew-1.13.0/lib/Release/x64/glew32.lib")
 

//#pragma comment(lib,"3rdParty/SDK/x64/cudart.lib")
//#pragma comment(lib,"3rdParty/SDK/x64/cuda.lib")
//#pragma comment(lib,"3rdParty/SDK/x64/cusparse.lib")
//#pragma comment(lib,"3rdParty/SDK/x64/cusolver.lib")
//#pragma comment(lib,"3rdParty/SDK/x64/cublas.lib")


//parameter

