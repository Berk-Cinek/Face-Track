#pragma once
#include "IFrameSource.h"


class CameraSource : public IFrameSource {
public: 
	CameraSource(int index, int height, int width);
	bool read(cv::Mat& frame)override;
private:
	cv::VideoCapture cap_;
};

