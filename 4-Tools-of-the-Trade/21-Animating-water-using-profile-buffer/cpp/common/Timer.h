#pragma once
#include <chrono>
using namespace std::chrono;

template <class T>
class Timer
{
public:
	std::string name;
	int cnt;
	T rt;
	high_resolution_clock::time_point tb;
	high_resolution_clock::time_point te;
	duration<T> dur;
	Timer() : rt((T)0.) {}
	Timer(const std::string func_name) : rt((T)0.), name(func_name) {}
	~Timer() {}
	void Begin()
	{
		tb = high_resolution_clock::now();
	}
	void End()
	{
		te = high_resolution_clock::now();
		dur = duration_cast<duration<T>>(te - tb);
		rt += dur.count();
		cnt += 1;
	}
	void Info()
	{
		std::cout << name << ": " << rt << "s" << std::endl;
	}
	void Ave_Info()
	{
		std::cout << name << ": " << rt / cnt << "s" << std::endl;
	}
	void Info(std::string funcName)
	{
		std::cout << funcName << ": " << rt << "s" << std::endl;
	}
	void TimeInfo(int frameNum)
	{
		std::cout << "After " << frameNum << " frames" << ": " << rt << "s" << std::endl;
	}
	void TimeFrameInfo(int frameNum)
	{
		std::cout << "At frame " << frameNum << ": " << dur.count() << "s" << std::endl;
	}
};