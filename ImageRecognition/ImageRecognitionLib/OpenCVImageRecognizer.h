#pragma once

#include <opencv2/dnn.hpp>

namespace ImageRecognitionLib
{
	public ref class OpenCVImageRecognizer sealed
	{
		cv::dnn::Net model;
		std::vector<std::wstring> classNames;
		OpenCVImageRecognizer(Platform::String^ modelFile, Platform::String^ classesFile);

	public:
		static Windows::Foundation::IAsyncOperation<OpenCVImageRecognizer^>^ OpenCVImageRecognizer::Create(Platform::String^ modelFile, Platform::String^ classesFile);
		Windows::Foundation::IAsyncOperation<Platform::String^>^ RecognizeObjectAsync(const Platform::Array<byte>^ bytes);
		uint32_t GetRequiredWidth();
		uint32_t GetRequiredHeight();
	};
}
