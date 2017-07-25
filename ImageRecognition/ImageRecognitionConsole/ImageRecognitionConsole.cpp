#include "stdafx.h"

#include "CNTKLibrary.h"

#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "..\Common\utils.inl"

class CNTKImageRecognizer sealed
{
	CNTK::DeviceDescriptor evalDevice = CNTK::DeviceDescriptor::UseDefaultDevice();
	CNTK::FunctionPtr model;
	CNTK::Variable inputVar;
	CNTK::NDShape inputShape;
	std::vector<std::wstring> classNames;
	CNTKImageRecognizer(const std::wstring& modelFile, const std::wstring& classesFile);
	std::wstring classifyImage(const uint8_t* image_data, size_t image_data_len);

public:
	static CNTKImageRecognizer Create(const std::wstring& modelFile, const std::wstring& classesFile);
	std::wstring RecognizeObjectAsync(const std::vector<uint8_t> bytes);
	uint32_t GetRequiredWidth();
	uint32_t GetRequiredHeight();
	uint32_t GetRequiredChannels();
};

using namespace cv;
using namespace CNTK;

std::wstring CNTKImageRecognizer::classifyImage(const uint8_t* image_data, size_t image_data_len)
{
	// Prepare the input vector and convert it to the correct color scheme (BBB ... GGG ... RRR)
	size_t resized_image_data_len = GetRequiredWidth() * GetRequiredHeight() * GetRequiredChannels();
	std::vector<uint8_t> image_data_array(resized_image_data_len);
	memcpy_s(image_data_array.data(), image_data_len, image_data, resized_image_data_len);
	std::vector<float> rawFeatures = get_features(image_data_array.data(), GetRequiredWidth(), GetRequiredHeight());

	// Prepare the input layer of the computation graph
	// Most of the work is putting rawFeatures into CNTK's data representation format
	std::unordered_map<CNTK::Variable, CNTK::ValuePtr> inputLayer = {};

	auto features = CNTK::Value::CreateBatch<float>(inputShape, rawFeatures, evalDevice, false);
	inputVar = model->Arguments()[0];
	inputLayer.insert({ inputVar, features });

	// Prepare the output layer of the computation graph
	// For this a NULL blob will be placed into the output layer 
	// so that CNTK can place its own datastructure there
	std::unordered_map<CNTK::Variable, CNTK::ValuePtr> outputLayer = {};
	CNTK::Variable outputVar = model->Output();
	CNTK::NDShape outputShape = outputVar.Shape();
	size_t possibleClasses = outputShape.Dimensions()[0];

	std::vector<float> rawOutputs(possibleClasses);
	auto outputs = CNTK::Value::CreateBatch<float>(outputShape, rawOutputs, evalDevice, false);
	outputLayer.insert({ outputVar, NULL });

	// Evaluate the image and extract the results (which will be a [ #classes x 1 x 1 ] tensor)
	model->Evaluate(inputLayer, outputLayer, evalDevice);

	CNTK::ValuePtr outputVal = outputLayer[outputVar];
	std::vector<std::vector<float>> resultsWrapper;
	std::vector<float> results;

	outputVal.get()->CopyVariableValueTo(outputVar, resultsWrapper);
	results = resultsWrapper[0];

	// Map the results to the string representation of the class
	int64_t image_class = find_class(results);
	return classNames.at(image_class);
}

uint32_t CNTKImageRecognizer::GetRequiredWidth()
{
	return (uint32_t)inputShape[0];
}

uint32_t CNTKImageRecognizer::GetRequiredHeight()
{
	return (uint32_t)inputShape[1];
}

uint32_t CNTKImageRecognizer::GetRequiredChannels()
{
	return (uint32_t)inputShape[2];
}

CNTKImageRecognizer::CNTKImageRecognizer(const std::wstring& modelFile, const std::wstring& classesFile)
{
	model = CNTK::Function::Load(modelFile, evalDevice);

	// List out all the outputs and their indexes
	// The probability output is usually listed as 'z' and is 
	// usually the last layer
	size_t z_index = model->Outputs().size() - 1;

	// Modify the in-memory model to use the z layer as the actual output
	auto z_layer = model->Outputs()[z_index];
	model = CNTK::Combine({ z_layer.Owner() });

	// Extract information about what the model accepts as input
	inputVar = model->Arguments()[0];
	// Shape contains image [width, height, depth] respectively
	inputShape = inputVar.Shape();

	// Load the class names
	classNames = read_class_names(classesFile);
}

CNTKImageRecognizer CNTKImageRecognizer::Create(const std::wstring& modelFile, const std::wstring& classesFile)
{
	return CNTKImageRecognizer(modelFile, classesFile);
}

std::wstring CNTKImageRecognizer::RecognizeObjectAsync(const std::vector<uint8_t> bytes)
{
/*
	// The data we've got is in RGBA format. We should convert it to BGR
	std::vector<uint8_t> rgb((bytes.size() / 4) * 3);
	const uint8_t* rgba = bytes.data();

	uint32_t i = 0;
	for (uint32_t j = 0; j < bytes.size();)
	{
		uint32_t r = j++;  // R
		uint32_t g = j++;  // G
		uint32_t b = j++;  // B
		uint32_t a = j++;  // A (skipped)

		rgb[i++] = rgba[r];
		rgb[i++] = rgba[g];
		rgb[i++] = rgba[b];
	}
*/
	auto image_class = classifyImage(bytes.data(), bytes.size());
	return image_class;
}

const int image_channels = 3;
const int feature_image_width = 224;
const int feature_image_height = 224;

int main()
{
	std::wstring model_file = L"..\\..\\resources\\models\\ResNet18_ImageNet_CNTK.model";

	if (!does_file_exist(model_file))
	{
		wprintf(L"Error: The model '%s' does not exist.\n", model_file.c_str());
		return 1;
	}

	std::wstring labels_file = L"..\\..\\resources\\models\\imagenet1000_clsid.txt";

	wprintf(L"Using model %s\n", model_file.c_str());

	std::wstring base_path = L"..\\..\\resources\\images";

	std::wstring classesFile = L"..\\..\\resources\\imagenet1000_clsid.txt";
	auto classNames = read_class_names(classesFile);

	std::vector<std::wstring> image_files = {
		L"timber-wolf.jpg",
		L"snow-leopard.jpg",
		L"cauliflower.jpg",
		L"broccoli.jpg"
	};

	auto cntkRecognizer = CNTKImageRecognizer::Create(model_file, labels_file);

	for (auto& image_file : image_files)
	{
		std::wstring image_file_path = base_path + L"\\" + image_file;
		if (!does_file_exist(image_file_path))
		{
			wprintf(L"Error: The image file '%s' does not exist.\n", model_file.c_str());
			return 1;
		}

		// Prepare image data
		Mat image = imread(wstrtostr(image_file_path), CV_LOAD_IMAGE_COLOR);
		Mat image_resized;

		resize(image, image_resized, Size(feature_image_width, feature_image_height));

		size_t resized_image_data_len = feature_image_width * feature_image_height * image_channels;
		std::vector<uint8_t> image_data_array(resized_image_data_len);

		memcpy(image_data_array.data(), image_resized.data, resized_image_data_len);

		wprintf(L"%s -> ", image_file.c_str());
		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

		auto objectName = cntkRecognizer.RecognizeObjectAsync(image_data_array);

		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

		wprintf(L"'%s'. Time elapsed: %lld ms.\n", objectName.c_str(), duration);

	}
	return 0;
}

