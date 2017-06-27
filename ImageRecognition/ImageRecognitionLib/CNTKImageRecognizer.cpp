#include "pch.h"
#include "CNTKImageRecognizer.h"

using namespace ImageRecognitionLib;
using namespace Platform;
using namespace Microsoft::MSR::CNTK;

const int image_channels = 3;
const int feature_image_width = 224;
const int feature_image_height = 224;

std::vector<std::string> read_class_names(std::string filename);
Platform::String^ StringFromCharPtr(const std::string str);


int64_t find_class(std::vector<float> outputs)
{
	return std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
}

std::vector<float> get_features(uint8_t* image_data_array, uint32_t reqWidth, uint32_t reqHeight)
{
	uint32_t size = reqWidth * reqHeight * 3;

	// BGR conversion to BBB..GGG..RRR
	std::vector<float> featuresLocal(size);
	float *pfeatures = featuresLocal.data();

	// convert BGR array to BBB...GGG...RRR array
	for (uint32_t c = 0; c < 3; c++) {
		for (uint32_t p = c; p < size; p = p + 3)
		{
			float v = image_data_array[p];
			*pfeatures++ = v;
		}
	}
	return featuresLocal;
}

std::string classify_image(CNTK::FunctionPtr model, std::vector<std::string>* classNames, const uint8_t* image_data, size_t image_data_len)
{
	size_t resized_image_data_len = feature_image_width * feature_image_height * image_channels;
	std::vector<uint8_t> image_data_array(resized_image_data_len);
	memcpy_s(image_data_array.data(), image_data_len, image_data, resized_image_data_len);
	std::vector<float> rawFeatures = get_features(image_data_array.data(), feature_image_height, feature_image_width);

	std::unordered_map<CNTK::Variable, CNTK::ValuePtr> inputLayer = {};
	CNTK::Variable inputVar = model->Arguments()[0];
	CNTK::NDShape inputShape = inputVar.Shape();
	auto device = CNTK::DeviceDescriptor::UseDefaultDevice();

	auto features = CNTK::Value::CreateBatch<float>(inputShape, rawFeatures, device, false);
	inputLayer.insert({ inputVar, features });

	std::unordered_map<CNTK::Variable, CNTK::ValuePtr> outputLayer = {};
	CNTK::Variable outputVar = model->Output();
	CNTK::NDShape outputShape = outputVar.Shape();
	int possibleClasses = outputShape.Dimensions()[0];

	std::vector<float> rawOutputs(possibleClasses);
	auto outputs = CNTK::Value::CreateBatch<float>(outputShape, rawOutputs, device, false);

	outputLayer.insert({ outputVar, NULL });

	model->Evaluate(inputLayer, outputLayer, device);

	CNTK::ValuePtr outputVal = outputLayer[outputVar];

	std::vector<std::vector<float>> resultsWrapper;
	std::vector<float> results;

	outputVal.get()->CopyVariableValueTo(outputVar, resultsWrapper);
	results = resultsWrapper[0];

	int64_t image_class = find_class(results);
	return classNames->at(image_class);
}

uint32_t CNTKImageRecognizer::GetRequiredWidth()
{
	return feature_image_width;
}

uint32_t CNTKImageRecognizer::GetRequiredHeight()
{
	return feature_image_height;
}

CNTKImageRecognizer::CNTKImageRecognizer(String^ modelFile, Platform::String^ classesFile)
{
	CNTK::DeviceDescriptor device = CNTK::DeviceDescriptor::CPUDevice();
	std::wstring w_str = std::wstring(modelFile->Data());
	model = CNTK::Function::Load(w_str, device);

	// List out all the outputs and their indexes
	// The probability output is usually listed as 'z' and is 
	// usually the last layer
	int z_index = model->Outputs().size() - 1;

	// Modify the in-memory model to use the z layer as the actual output
	auto z_layer = model->Outputs()[z_index];
	model = CNTK::Combine({ z_layer.Owner() });

	// Load the class names
	w_str = std::wstring(classesFile->Data());
	std::string s_str = std::string(w_str.begin(), w_str.end());
	classNames = read_class_names(s_str);
}

Windows::Foundation::IAsyncOperation<CNTKImageRecognizer^>^ CNTKImageRecognizer::Create(Platform::String^ modelFile, Platform::String^ classesFile)
{
	return concurrency::create_async([=] {
		return ref new CNTKImageRecognizer(modelFile, classesFile);
	});
}

Windows::Foundation::IAsyncOperation<Platform::String^>^ CNTKImageRecognizer::RecognizeObjectAsync(const Platform::Array<byte>^ bytes)
{
	return concurrency::create_async([=] {
		// The data we've got is in RGBA format. We should convert it to BGR
		std::vector<uint8_t> rgb((bytes->Length / 4) * 3);
		uint8_t* rgba = bytes->Data;

		uint32_t i = 0;
		for (uint32_t j = 0; j < bytes->Length;)
		{
			uint32_t r = j++;  // R
			uint32_t g = j++;  // G
			uint32_t b = j++;  // B
			uint32_t a = j++;  // A (skipped)

			rgb[i++] = rgba[r];
			rgb[i++] = rgba[g];
			rgb[i++] = rgba[b];
		}

		auto image_class = classify_image(model, &classNames, rgb.data(), rgb.size());
		return StringFromCharPtr(image_class);
	});
}