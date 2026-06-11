#include "UdpSource.h"
#include <opencv2/videoio.hpp>

UdpSource::UdpSource(const std::string& url, int height, int width) : cap_(url) {
	if (!cap_.isOpened())
		throw std::runtime_error("UdpSouce: could not connect to device " + url);

	cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
	cap_.set(cv::CAP_PROP_FRAME_WIDTH, width);
}

bool UdpSource::read(cv::Mat& frame) {
	return cap_.read(frame) && !frame.empty();
}