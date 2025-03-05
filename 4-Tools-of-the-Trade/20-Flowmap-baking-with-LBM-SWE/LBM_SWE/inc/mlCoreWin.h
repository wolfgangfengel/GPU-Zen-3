#pragma once
#ifndef MLCOREWIN_H
#define MLCOREWIN_H


#ifndef MLFUNC_TYPE
#ifdef MLCUDA_DEVICE	
#define MLFUNC_TYPE __host__ __device__
#else
#define MLFUNC_TYPE
#endif//MLCUDA_DEVICE
#endif//MLFUNC_TYPE

#define REAL float

#endif //MLCOREWIN_H