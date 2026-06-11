#include "CameraSource.h"
#include <opencv2/videoio.hpp>

CameraSource::CameraSource(int index, int height, int width) : cap_(index) {
	if (!cap_.isOpened())
		throw std::runtime_error("CameraSource: could not open device" + std::to_string(index));

	cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 640);
	cap_.set(cv::CAP_PROP_FRAME_WIDTH, 480);
}

bool CameraSource::read(cv::Mat& frame) {
	return cap_.read(frame) && !frame.empty();
}