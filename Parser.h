#pragma once
#include <vector>
#include "BackendHelperModels.h"
#include <onnxruntime_cxx_api.h>

namespace Parser {
    std::vector<FaceData> parse_scrfd_ort(const std::vector<Ort::Value>& outputs, int img_w, int img_h, float conf_thresh = 0.5f, float nms_thresh = 0.45f);

    std::vector<cv::Point3d> parse_1k3d68_ort(const std::vector<Ort::Value>& outputs, const cv::Mat& M);
}
