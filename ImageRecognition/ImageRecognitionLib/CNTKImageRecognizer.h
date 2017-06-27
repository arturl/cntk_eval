#pragma once

#include "CNTKLibrary.h"

namespace ImageRecognitionLib
{
	public ref class CNTKImageRecognizer sealed
	{
		CNTK::FunctionPtr model;
		std::vector<std::string> classNames;
		CNTKImageRecognizer(Platform::String^ modelFile, Platform::String^ classesFile);

	public:
		static Windows::Foundation::IAsyncOperation<CNTKImageRecognizer^>^ CNTKImageRecognizer::Create(Platform::String^ modelFile, Platform::String^ classesFile);
		Windows::Foundation::IAsyncOperation<Platform::String^>^ RecognizeObjectAsync(const Platform::Array<byte>^ bytes);
		uint32_t GetRequiredWidth();
		uint32_t GetRequiredHeight();
	};
}
