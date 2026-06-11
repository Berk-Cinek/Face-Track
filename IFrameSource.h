#pragma once
#include <opencv2/core.hpp>

class IFrameSource {
public:
	virtual ~IFrameSource() = default;
	virtual bool read(cv::Mat& frame) = 0;
};