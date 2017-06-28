#include "pch.h"
#include "OpenCVImageRecognizer.h"

#include <opencv2/dnn.hpp> 
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace ImageRecognitionLib;
using namespace Platform;

const cv::String tfInBlobName = ".input";
const cv::String tfOutBlobName = "softmax2";

const int image_channels = 3;
const int feature_image_width = 224;
const int feature_image_height = 224;

std::vector<std::string> read_class_names(std::string filename);
Platform::String^ StringFromCharPtr(const std::string str);

// From http://docs.opencv.org/trunk/d5/de7/tutorial_dnn_googlenet.html
void getMaxClass(const cv::Mat &probBlob, int *classId, double *classProb)
{
	cv::Mat probMat = probBlob.reshape(1, 1);
	cv::Point classNumber;

	minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
	*classId = classNumber.x;
}

uint32_t OpenCVImageRecognizer::GetRequiredWidth()
{
	return feature_image_width;
}

uint32_t OpenCVImageRecognizer::GetRequiredHeight()
{
	return feature_image_height;
}

std::string classify_image(cv::dnn::Net* model, std::vector<std::string>* classNames, uint8_t* image_data, size_t image_data_len)
{
	// Prepare the input layer of the computation graph
	cv::Mat img = cv::Mat(feature_image_width, feature_image_height, CV_8UC3, (unsigned*)image_data);
	cv::dnn::Blob inputBlob = cv::dnn::Blob::fromImages(img);
	model->setBlob(tfInBlobName, inputBlob);

	// Evaluate the image and extract the results
	model->forward();

	cv::dnn::Blob prob = model->getBlob(tfOutBlobName);
	cv::Mat& result = prob.matRef();

	// Map the results to the string representation of the class
	int classId;
	double classProb;
	getMaxClass(result, &classId, &classProb);

	return classNames->at(classId);
}

OpenCVImageRecognizer::OpenCVImageRecognizer(String^ modelFile, Platform::String^ classesFile)
{
	std::wstring w_str = std::wstring(modelFile->Data());
	std::string s_str = std::string(w_str.begin(), w_str.end());
	cv::String c_str = cv::String(s_str);
	cv::Ptr<cv::dnn::Importer> importer;
	importer = cv::dnn::createTensorflowImporter(c_str);
	importer->populateNet(model);
	importer.release();

	// Load the class names
	w_str = std::wstring(classesFile->Data());
	s_str = std::string(w_str.begin(), w_str.end());
	classNames = read_class_names(s_str);
}

Windows::Foundation::IAsyncOperation<OpenCVImageRecognizer^>^ OpenCVImageRecognizer::Create(Platform::String^ modelFile, Platform::String^ classesFile)
{
	return concurrency::create_async([=] {
		return ref new OpenCVImageRecognizer(modelFile, classesFile);
	});
}

Windows::Foundation::IAsyncOperation<Platform::String^>^ OpenCVImageRecognizer::RecognizeObjectAsync(const Platform::Array<byte>^ bytes)
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

		auto image_class = classify_image(&model, &classNames, rgb.data(), rgb.size());
		return StringFromCharPtr(image_class);
	});
}