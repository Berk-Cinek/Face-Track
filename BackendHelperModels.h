#pragma once
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

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

struct FacePoseController {
    bool has_face = false;
    int lost_frames = 0;
    PoseState last_accepted;
    FaceData last_face;
    bool pose_valid = false;

    void initialize(const std::vector<FaceData>& faces, cv::Mat rvec, cv::Mat tvec) {
        if (!pose_valid && !faces.empty()) {
            smooth(last_accepted, rvec, tvec);
            last_face = faces[0];
            has_face = true;
            lost_frames = 0;
            pose_valid = true;
        }
    }
    void update(const std::vector<FaceData>& faces, cv::Mat rvec_raw, cv::Mat tvec_raw) {

        if (!faces.empty() && gate(last_accepted, rvec_raw, tvec_raw)) {
            smooth(last_accepted, rvec_raw, tvec_raw);
            last_face = faces[0];
            has_face = true;
            lost_frames = 0;
            pose_valid = true;
        }
        else {
            lost_frames++;
            if (lost_frames > 10) {
                has_face = false;
                pose_valid = false;
            }
        }
    }

    bool gate(PoseState& accepted_last, const cv::Mat& rvec_raw, const cv::Mat& tvec_raw) {

        if (accepted_last.rvec.empty()) {
            return true; // first valid pose, accept
        }

        const double threshold_rotation = 0.35; // rad
        const double threshold_distance = 0.40; // %

        cv::Mat R_raw, R_last;
        cv::Rodrigues(rvec_raw, R_raw);
        cv::Rodrigues(accepted_last.rvec, R_last);

        cv::Mat R_delta = R_raw * R_last.t();
        double trace = cv::trace(R_delta)[0];
        double cos_theta = (trace - 1.0) / 2.0;
        cos_theta = std::clamp(cos_theta, -1.0, 1.0);
        double rotation_angle = std::acos(cos_theta);

        double tz = tvec_raw.at<double>(2, 0);
        double tz_last = accepted_last.tvec.at<double>(2, 0);

        double denom = std::max(1e-6, std::max(tz, tz_last));
        double distance_delta = std::abs(tz - tz_last) / denom;

        if (rotation_angle <= threshold_rotation && distance_delta <= threshold_distance) {
            return true;
        }

        spdlog::info("Gating triggered (frame dropped). rot(rad)={:.3f}, dist(%)={:.3f}",
            rotation_angle, distance_delta);
        return false;
    }

    void smooth(PoseState& last_accepted, cv::Mat rvec_raw, cv::Mat tvec_raw) {

        if (last_accepted.rvec.empty()) {
            last_accepted.rvec = rvec_raw.clone();
            last_accepted.tvec = tvec_raw.clone();
        }

        cv::Mat R_raw;
        cv::Mat R_last;
        cv::Mat rvec_delta;
        cv::Mat R_step;

        cv::Rodrigues(rvec_raw, R_raw);
        cv::Rodrigues(last_accepted.rvec, R_last);

        cv::Mat R_delta = R_raw * R_last.t();

        cv::Rodrigues(R_delta, rvec_delta);

        double alpha_rot = 0.8;
        rvec_delta = rvec_delta * (1.0 - alpha_rot);

        cv::Rodrigues(rvec_delta, R_step);
        cv::Mat R_new = R_step * R_last;
        cv::Rodrigues(R_new, last_accepted.rvec);

        const double alpha_t = 0.85;
        last_accepted.tvec = alpha_t * last_accepted.tvec + (1.0 - alpha_t) * tvec_raw;

    };

};

