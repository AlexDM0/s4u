#pragma once

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <windows.h>

class profiler {
public:
	unsigned __int64 frequency;
	unsigned __int64 startTime;
	unsigned __int64 endTime;
	double timerFrequency;
	bool active;
	std::string description;

	profiler() {
		frequency = 0;
		endTime = 0;
		startTime = 0;
		description = "";
		active = true;

		QueryPerformanceFrequency((LARGE_INTEGER*)&frequency);
		timerFrequency = (1.0/frequency);
	}

	profiler(bool enabled) {
		frequency = 0;
		endTime = 0;
		startTime = 0;
		description = "";
		active = enabled;

		QueryPerformanceFrequency((LARGE_INTEGER*)&frequency);
		timerFrequency = (1.0/frequency);
	}

	void start() {
		if (active) {
			QueryPerformanceCounter((LARGE_INTEGER *)&startTime);
		}
	}

	void start(std::string description) {
		if (active) {
			this->description = description;
			start();
		}
	}

	void end() {
		if (active) {
			QueryPerformanceCounter((LARGE_INTEGER *)&endTime);
			double timeDifferenceInMilliseconds = ((endTime-startTime) * timerFrequency);
			std::cout << description << ": " << timeDifferenceInMilliseconds << std::endl;
		}
	}

	void end(std::string description) {
		if (active) {
			this->description = description;
			end();
		}
	}
};








