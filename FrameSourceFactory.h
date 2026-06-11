#pragma once
#include "IFrameSource.h"

enum class SourceType{ Webcam, Usb, Udp};

std::unique_ptr<IFrameSource> makeFrameSource(SourceType type, const std::string& config);