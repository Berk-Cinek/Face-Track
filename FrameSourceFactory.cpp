#include "FrameSourceFactory.h"
#include "CameraSource.h"
#include "UdpSource.h"

std::unique_ptr<IFrameSource> makeFrameSource(SourceType type, const std::string& config) {
	switch (type)
	{
	case SourceType::Webcam:
		return std::make_unique<CameraSource>(0, 640, 480);
	case SourceType::Usb:
		return std::make_unique<CameraSource>(std::stoi(config), 640, 480);
	case SourceType::Udp:
		return std::make_unique<UdpSource>(config);
	default:
		throw std::runtime_error("makeFrameSource: unkown source type");
	}
}