#include "pch.h"
#include <fstream>

Platform::String^ StringFromCharPtr(const std::string str)
{
	std::wstring wid_str = std::wstring(str.begin(), str.end());
	const wchar_t* w_char = wid_str.c_str();
	return ref new Platform::String(w_char);
}

// From http://docs.opencv.org/trunk/d5/de7/tutorial_dnn_googlenet.html
std::vector<std::string> read_class_names(std::string filename)
{
	std::vector<std::string> classNames;
	std::ifstream fp(filename);
	if (!fp.is_open())
	{
		return classNames;
	}
	std::string name;
	while (!fp.eof())
	{
		std::getline(fp, name);
		if (name.length())
			classNames.push_back(name.substr(name.find(' ') + 1));
	}
	fp.close();
	return classNames;
}