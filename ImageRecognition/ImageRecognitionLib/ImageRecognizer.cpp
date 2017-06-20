#include "pch.h"
#include "ImageRecognizer.h"

using namespace ImageRecognitionLib;
using namespace Platform;
using namespace Microsoft::MSR::CNTK;

const int image_channels = 3;
const int feature_image_width = 224;
const int feature_image_height = 224;

typedef std::pair<std::wstring, std::vector<float>*> MapEntry;
typedef std::map<std::wstring, std::vector<float>*> Layer;

Platform::String^ StringFromCharPtr(const std::string str)
{
	std::wstring wid_str = std::wstring(str.begin(), str.end());
	const wchar_t* w_char = wid_str.c_str();
	return ref new Platform::String(w_char);
}

std::vector<float> get_features(uint8_t* image_data_array, uint32_t reqWidth, uint32_t reqHeight)
{
	uint32_t size = reqWidth * reqHeight * 3;

	// BGR conversion to BBB..GGG..RRR
	std::vector<float> featuresLocal;

	// convert BGR array to BBB...GGG...RRR array
	for (uint32_t c = 0; c < 3; c++) {
		for (uint32_t p = c; p < size; p = p + 3)
		{
			float v = image_data_array[p];
			featuresLocal.push_back(v);
		}
	}
	return featuresLocal;
}

IEvaluateModel<float> * prepare_model(const std::string& model_file)
{
	IEvaluateModel<float> * model;
	GetEvalF(&model);

	// Load model with desired outputs
	std::string networkConfiguration;
	networkConfiguration += "modelPath=\"" + model_file + "\"";
	model->CreateNetwork(networkConfiguration);

	return model;
}

int64_t find_class(std::vector<float> outputs)
{
	return std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
}

std::string map_image_class_number_to_image_class_string(int64_t image_class);

bool does_file_exist(std::string  file_name)
{
	return std::experimental::filesystem::exists(file_name);
}

std::string classify_image(IEvaluateModel<float> * model, const uint8_t* image_data, size_t image_data_len)
{
	// Get the model's layers dimensions
	std::map<std::wstring, size_t> inDims;
	std::map<std::wstring, size_t> outDims;
	model->GetNodeDimensions(inDims, NodeGroup::nodeInput);
	model->GetNodeDimensions(outDims, NodeGroup::nodeOutput);

	// Prepare image data

	size_t resized_image_data_len = feature_image_width * feature_image_height * image_channels;
	std::vector<uint8_t> image_data_array(resized_image_data_len);

	memcpy_s(image_data_array.data(), image_data_len, image_data, resized_image_data_len);

	// Make image features from image data

	auto features = get_features(image_data_array.data(), feature_image_height, feature_image_width);

	auto inputLayerName = inDims.begin()->first;
	Layer inputLayer;
	inputLayer.insert(MapEntry(inputLayerName, &features));

	Layer outputLayer;
	auto outputLayerName = outDims.begin()->first;
	std::vector<float> outputs;
	outputLayer.insert(MapEntry(outputLayerName, &outputs));

	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

	model->Evaluate(inputLayer, outputLayer);

	int64_t image_class = find_class(outputs);

	std::string image_class_str = map_image_class_number_to_image_class_string(image_class);

	return image_class_str;
}


ImageRecognizer::ImageRecognizer(String^ modelFile)
{
	GetEvalF(&m_model);

	std::wstring w_str = std::wstring(modelFile->Data());
	std::string s_str = std::string(w_str.begin(), w_str.end());

	// Load model with desired outputs
	std::string networkConfiguration;
	networkConfiguration = "modelPath=\"" + s_str + "\"";
	m_model->CreateNetwork(networkConfiguration);
}

uint32_t ImageRecognizer::GetRequiredWidth()
{
	return feature_image_width;
}

uint32_t ImageRecognizer::GetRequiredHeight()
{
	return feature_image_height;
}

Platform::String^ ImageRecognizer::RecognizeObject(const Platform::Array<byte>^ bytes)
{
	// The data we've got is in RGBA format. We should convert it to RGB
	std::vector<uint8_t> rgb((bytes->Length / 4) * 3);
	uint8_t* rgba = bytes->Data;

	uint32_t i = 0;
	for (uint32_t j = 0; j < bytes->Length;)
	{
		rgb[i++] = rgba[j++]; 	// R
		rgb[i++] = rgba[j++]; 	// G
		rgb[i++] = rgba[j++]; 	// B
		j++;					// A (skipped)
	}

	auto image_class = classify_image(m_model, rgb.data(), rgb.size());
	return StringFromCharPtr(image_class);
}