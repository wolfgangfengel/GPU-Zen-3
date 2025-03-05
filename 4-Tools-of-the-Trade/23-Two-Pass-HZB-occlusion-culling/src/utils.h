#pragma once

std::vector<char> readFile(
	const char* _pFilePath);

m4 getInfinitePerspectiveMatrix(
	f32 _fov,
	f32 _aspect,
	f32 _near);

u32 roundUpToPowerOfTwo(
	f32 _value);
