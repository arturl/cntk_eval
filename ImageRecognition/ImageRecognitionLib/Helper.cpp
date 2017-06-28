#include "pch.h"
#include <fstream>

/**
	Convert a std::str to a Platform::String^

	@param str The std::string to convert
	@return A Platform::String^ representation of str
*/
Platform::String^ StringFromCharPtr(const std::string str)
{
	std::wstring wid_str = std::wstring(str.begin(), str.end());
	const wchar_t* w_char = wid_str.c_str();
	return ref new Platform::String(w_char);
}

/**
	Read in a text file where each line is a class into a vector.
	From http://docs.opencv.org/trunk/d5/de7/tutorial_dnn_googlenet.html

	@param filename The path to the text file
	@return A vector where each entry represents the class name given
	the output of a machine learning classifier.
*/
std::vector<std::string> read_class_names(const std::string filename)
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