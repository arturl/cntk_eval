#include "stdafx.h"

#include "Eval.h"

#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace Microsoft::MSR::CNTK;

const int image_channels = 3;
const int feature_image_width = 224;
const int feature_image_height = 224;

typedef std::pair<std::wstring, std::vector<float>*> MapEntry;
typedef std::map<std::wstring, std::vector<float>*> Layer;

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

int main()
{
	std::string model_file = "..\\..\\resources\\models\\ResNet18_ImageNet_CNTK.model";

	if (!does_file_exist(model_file))
	{
		fprintf(stderr, "Error: The model '%s' does not exist.\n", model_file.c_str());
		return 1;
	}

	printf("Using model %s\n", model_file.c_str());

	IEvaluateModel<float> *model = prepare_model(model_file);

	std::string base_path = "..\\..\\resources\\images";

	std::vector<std::string> image_files = {
		"timber-wolf.jpg",
		"snow-leopard.jpg",
		"cauliflower.jpg",
		"broccoli.jpg"
	};

	for (auto& image_file : image_files)
	{
		std::string image_file_path = base_path + "\\" + image_file;
		if (!does_file_exist(image_file_path))
		{
			fprintf(stderr, "Error: The image file '%s' does not exist.\n", model_file.c_str());
			return 1;
		}

		// Get the model's layers dimensions
		std::map<std::wstring, size_t> inDims;
		std::map<std::wstring, size_t> outDims;
		model->GetNodeDimensions(inDims, NodeGroup::nodeInput);
		model->GetNodeDimensions(outDims, NodeGroup::nodeOutput);

		// Prepare image data

		Mat image = imread(image_file_path, CV_LOAD_IMAGE_COLOR);
		Mat image_resized;

		resize(image, image_resized, Size(feature_image_width, feature_image_height));

		size_t resized_image_data_len = feature_image_width * feature_image_height * image_channels;
		std::vector<uint8_t> image_data_array(resized_image_data_len);

		memcpy(image_data_array.data(), image_resized.data, resized_image_data_len);

		// Make image features from image data

		auto features = get_features(image_data_array.data(), feature_image_height, feature_image_width);

		auto inputLayerName = inDims.begin()->first;
		Layer inputLayer;
		inputLayer.insert(MapEntry(inputLayerName, &features));

		Layer outputLayer;
		auto outputLayerName = outDims.begin()->first;
		std::vector<float> outputs;
		outputLayer.insert(MapEntry(outputLayerName, &outputs));

		printf("%s -> ", image_file.c_str());
		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

		model->Evaluate(inputLayer, outputLayer);

		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

		int64_t image_class = find_class(outputs);

		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

		std::string image_class_str = map_image_class_number_to_image_class_string(image_class);

		printf("%s. time elapsed: %lld\n", image_class_str.c_str(), duration);
	}
	return 0;
}

