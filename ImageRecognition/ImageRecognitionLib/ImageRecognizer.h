#pragma once

#include "Eval.h"

namespace ImageRecognitionLib
{
	public ref class ImageRecognizer sealed
	{
		Microsoft::MSR::CNTK::IEvaluateModel<float> * m_model;
		ImageRecognizer(Platform::String^ modelFile);

	public:
		
		static Windows::Foundation::IAsyncOperation<ImageRecognizer^>^ Create(Platform::String^ modelFile);
		Windows::Foundation::IAsyncOperation<Platform::String^>^ RecognizeObjectAsync(const Platform::Array<byte>^ bytes);

		uint32_t GetRequiredWidth();
		uint32_t GetRequiredHeight();
	};
}
