#include "stdafx.h"

#include "Eval.h"
#include "CNTKLibrary.h"

#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "..\Common\utils.inl"

using namespace cv;
using namespace Microsoft::MSR::CNTK;

const int image_channels = 3;
const int feature_image_width = 224;
const int feature_image_height = 224;

typedef std::pair<std::wstring, std::vector<float>*> MapEntry;
typedef std::map<std::wstring, std::vector<float>*> Layer;

int main()
{
	std::wstring model_file = L"..\\..\\resources\\models\\ResNet18_ImageNet_CNTK.model";

	if (!does_file_exist(model_file))
	{
		wprintf(L"Error: The model '%s' does not exist.\n", model_file.c_str());
		return 1;
	}

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

	CNTK::DeviceDescriptor evalDevice = CNTK::DeviceDescriptor::UseDefaultDevice();
	auto model = CNTK::Function::Load(model_file, evalDevice);

	// List out all the outputs and their indexes
	// The probability output is usually listed as 'z' and is 
	// usually the last layer
	size_t z_index = model->Outputs().size() - 1;

	// Modify the in-memory model to use the z layer as the actual output
	auto z_layer = model->Outputs()[z_index];
	model = CNTK::Combine({ z_layer.Owner() });

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

		// Make image features from image data
		std::vector<float> rawFeatures = get_features(image_data_array.data(), feature_image_width, feature_image_height);

		// Extract information about what the model accepts as input
		auto inputVar = model->Arguments()[0];
		// Shape contains image [width, height, depth] respectively
		auto inputShape = inputVar.Shape();

		auto features = CNTK::Value::CreateBatch<float>(inputShape, rawFeatures, evalDevice, false);

		std::unordered_map<CNTK::Variable, CNTK::ValuePtr> inputLayer = {};
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

		wprintf(L"%s -> ", image_file.c_str());
		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

		// Evaluate the image and extract the results (which will be a [ #classes x 1 x 1 ] tensor)
		model->Evaluate(inputLayer, outputLayer, evalDevice);

		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

		CNTK::ValuePtr outputVal = outputLayer[outputVar];
		std::vector<std::vector<float>> resultsWrapper;
		std::vector<float> results;

		outputVal.get()->CopyVariableValueTo(outputVar, resultsWrapper);
		results = resultsWrapper[0];

		// Map the results to the string representation of the class
		int64_t image_class = find_class(results);

		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

		std::wstring image_class_str = classNames[image_class];

		wprintf(L"'%s'. Time elapsed: %lld ms.\n", image_class_str.c_str(), duration);

	}
	return 0;
}

