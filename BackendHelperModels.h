#pragma once
#include <vector>
#include <opencv2/core.hpp>            
#include <opencv2/video/tracking.hpp>

struct FaceData {
    cv::Rect2d bounding_box;
    std::vector<cv::Point2d> landmarks;
    float confidence = 0.0f;

};

struct letterBoxInfo {
    float scale;
    int pad_x;
    int pad_y;
    int dst_w;
    int dst_h;
};


struct PoseState {
    cv::Mat rvec;
    cv::Mat tvec;
};


class PoseKalmanFilter {
private:

    cv::KalmanFilter channels[6]; // rx, ry, rz, tx, ty, tz

    void initChannel(cv::KalmanFilter& kf);

    float updateChannel(cv::KalmanFilter& kf, float measurement);

public:
    PoseKalmanFilter();

    void filter(const cv::Mat& rvec_in, const cv::Mat& tvec_in, cv::Mat& rvec_out, cv::Mat& tvec_out);

};

class FacePoseController {
private:
    bool has_face = false;
    int lost_frames = 0;
    PoseState last_accepted;
    FaceData last_face;
    bool pose_valid = false;

public:
    void initialize(const std::vector<FaceData>& faces, cv::Mat rvec, cv::Mat tvec);

    void update(const std::vector<FaceData>& faces, cv::Mat rvec_raw, cv::Mat tvec_raw);
};

