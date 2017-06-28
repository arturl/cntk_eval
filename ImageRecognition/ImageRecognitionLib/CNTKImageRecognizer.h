#pragma once

#include "CNTKLibrary.h"

namespace ImageRecognitionLib
{
	public ref class CNTKImageRecognizer sealed
	{
		CNTK::DeviceDescriptor evalDevice = CNTK::DeviceDescriptor::UseDefaultDevice();
		CNTK::FunctionPtr model;
		CNTK::Variable inputVar;
		CNTK::NDShape inputShape;
		std::vector<std::string> classNames;
		CNTKImageRecognizer(Platform::String^ modelFile, Platform::String^ classesFile);
		std::string classifyImage(const uint8_t* image_data, size_t image_data_len);

	public:
		static Windows::Foundation::IAsyncOperation<CNTKImageRecognizer^>^ CNTKImageRecognizer::Create(Platform::String^ modelFile, Platform::String^ classesFile);
		Windows::Foundation::IAsyncOperation<Platform::String^>^ RecognizeObjectAsync(const Platform::Array<byte>^ bytes);
		uint32_t GetRequiredWidth();
		uint32_t GetRequiredHeight();
		uint32_t GetRequiredChannels();
	};
}
