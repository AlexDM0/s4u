#pragma once


template <typename T>
std::string toStr(T a) {
	std::stringstream concatenatedStr;
	concatenatedStr.str("");
	concatenatedStr << a;
	return concatenatedStr.str();
}

template <typename T1, typename T2>
std::string addStr(T1 a, T2 b) {
	std::stringstream concatenatedStr;
	concatenatedStr.str("");
	concatenatedStr << a << b;
	return concatenatedStr.str();
}

template <typename T1, typename T2, typename T3>
std::string addStr(T1 a, T2 b, T3 c) {
	std::stringstream concatenatedStr;
	concatenatedStr.str("");
	concatenatedStr << a << b << c;
	return concatenatedStr.str();
}

template <typename T1, typename T2, typename T3, typename T4>
std::string addStr(T1 a, T2 b, T3 c, T4 d) {
	std::stringstream concatenatedStr;
	concatenatedStr.str("");
	concatenatedStr << a << b << c << d;
	return concatenatedStr.str();
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
std::string addStr(T1 a, T2 b, T3 c, T4 d, T5 e) {
	std::stringstream concatenatedStr;
	concatenatedStr.str("");
	concatenatedStr << a << b << c << d << e;
	return concatenatedStr.str();
}

