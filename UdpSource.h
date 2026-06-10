#pragma once
#include "IFrameSource.h"

class UdpSource : public IFrameSource {
public:
	UdpSource(const std::string& url);
	bool read(cv::Mat& frame) override;
private:
	cv::VideoCapture cap_;
};

