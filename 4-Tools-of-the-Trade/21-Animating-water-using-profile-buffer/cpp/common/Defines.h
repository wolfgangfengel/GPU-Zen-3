#pragma once

using real = float;

// physical constants
#define PI				3.14159265359f
#define TAU				6.28318530718f
#define GRAVITY			9.81f

// profile buffer constants
#define DIR_NUM			32
#define SEG_PER_DIR		8
#define SEED			4023432
#define PB_RESOLUTION	4096
#define WAVE_DIM		4
#define FINE_DIR_NUM	(DIR_NUM * SEG_PER_DIR)

// scene setup
#define GRID_W			1024
#define GRID_L			1024
#define DOMAIN_SCALE	200.f
#define DX				(DOMAIN_SCALE / GRID_W)

// GUI setup
#define WINDOW_WIDTH	1920
#define WINDOW_HEIGHT	1080
#define SHOW_CAMERA_INFO 
#define EXPORT_IMG
#ifdef EXPORT_IMG
#define EXPORT_STEPS	7500
#define STEPS_PER_FRAME 10
#endif
