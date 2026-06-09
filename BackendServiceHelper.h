#pragma once
#include <vector>
#include <opencv2/core.hpp>
#include "BackendHelperModels.h"

namespace LetterBoxGeometry {
    cv::Point2d unletter_point(const cv::Point2d& pad, const letterBoxInfo& letterbox);

    cv::Rect2d unletter_rect(const cv::Rect2d& r, const letterBoxInfo& letterbox);

    letterBoxInfo letterbox(const cv::Mat& src, cv::Mat& dst, int net_w, int net_h);

    std::vector<FaceData> unletterbox_faces(const std::vector<FaceData>& faces640, const letterBoxInfo& lb, int orig_w, int orig_h);
};

namespace ModelInput{

    void pack_nchw_rgb(const cv::Mat& img, float* out, float mean, float scale);

    cv::Mat cropFaceFor1k3d68(const cv::Mat& frame, const FaceData& face, cv::Mat& dst, float* input_buffer);
};