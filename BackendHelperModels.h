#pragma once
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
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

class PoseKalamanFilter {
private:
    cv::KalmanFilter channels[6]; // rx, ry, rz, tx, ty, tz

    void initChannel(cv::KalmanFilter& kf) {
        kf.init(2, 1, 0, CV_32F);
        kf.transitionMatrix = (cv::Mat_<float>(2, 2) << 1, 1, 0, 1);
        kf.measurementMatrix = (cv::Mat_<float>(1, 2) << 1, 0);

        cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-4)); //Q tuning
        cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-2)); //R tuning
        cv::setIdentity(kf.errorCovPost, cv::Scalar(1.0));
    }

    float updateChannel(cv::KalmanFilter& kf, float measurement)
    {
        kf.predict();
        cv::Mat measure = (cv::Mat_<float>(1, 1) << measurement);
        cv::Mat corrected = kf.correct(measure);
        return corrected.at<float>(0);
    }

public:

    PoseKalamanFilter() {
        for (auto& kf : channels)
            initChannel(kf);
    }

    void filter(const cv::Mat& rvec_in, const cv::Mat& tvec_in, cv::Mat& rvec_out, cv::Mat& tvec_out) {
        double rv[3], tv[3];
        for (int i = 0; i < 3; i++)
        {
            rv[i] = updateChannel(channels[i], (float)rvec_in.at<double>(i));
            tv[i] = updateChannel(channels[i + 3], (float)tvec_in.at<double>(i));
        }
        rvec_out = (cv::Mat_<double>(3, 1) << rv[0], rv[1], rv[2]);
        tvec_out = (cv::Mat_<double>(3, 1) << tv[0], tv[1], tv[2]);
    }

};

struct FacePoseController {
    bool has_face = false;
    int lost_frames = 0;
    PoseState last_accepted;
    FaceData last_face;
    bool pose_valid = false;

    void initialize(const std::vector<FaceData>& faces, cv::Mat rvec, cv::Mat tvec) {
        if (!pose_valid && !faces.empty()) {
          last_accepted.rvec = rvec.clone();
          last_accepted.tvec = tvec.clone();
          last_face = faces[0];
          has_face = true;
          lost_frames = 0;
          pose_valid = true;
        }
    }


    void update(const std::vector<FaceData>& faces, cv::Mat rvec_raw, cv::Mat tvec_raw) {

        if (!faces.empty()) {
            last_accepted.rvec = rvec_raw.clone();
            last_accepted.tvec = tvec_raw.clone();
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

    // estimate 3x4 affine matrix mapping mean shape X onto predicted landmarks Y
    cv::Mat estimateAffine3Dto3D(const cv::Mat& X, const cv::Mat& Y) {
        int n = X.rows;
        cv::Mat ones = cv::Mat::ones(n, 1, CV_64F);
        cv::Mat X_homo;
        cv::hconcat(X, ones, X_homo);     //[68, 4]

        cv::Mat P_T;
        cv::solve(X_homo, Y, P_T, cv::DECOMP_SVD); //least squares
        return P_T.t();        //[3, 4]
    }

    // decompose affine matrix into scale, rotation, translation
    void P2sRt(const cv::Mat& P, double& s, cv::Mat& R, cv::Mat& t) {
        t = P.col(3).clone();              // [3, 1]

        cv::Mat R1 = P.row(0).colRange(0, 3);
        cv::Mat R2 = P.row(1).colRange(0, 3);

        double n1 = cv::norm(R1);
        double n2 = cv::norm(R2);
        s = (n1 + n2) / 2.0;

        cv::Mat r1 = R1 / n1;
        cv::Mat r2 = R2 / n2;
        cv::Mat r3 = r1.cross(r2);

        R = cv::Mat(3, 3, CV_64F);
        r1.copyTo(R.row(0));
        r2.copyTo(R.row(1));
        r3.copyTo(R.row(2));
    }

    // rotation matrix → pitch, yaw, roll in degrees
    void matrix2angle(const cv::Mat& R, double& pitch, double& yaw, double& roll) {
        double sy = std::sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) +
            R.at<double>(1, 0) * R.at<double>(1, 0));

        if (sy > 1e-6) {
            pitch = std::atan2(R.at<double>(2, 1), R.at<double>(2, 2));
            yaw = std::atan2(-R.at<double>(2, 0), sy);
            roll = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0));
        }
        else {
            pitch = std::atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
            yaw = std::atan2(-R.at<double>(2, 0), sy);
            roll = 0.0;
        }

        const double deg = 180.0 / CV_PI;
        pitch *= deg;
        yaw *= deg;
        roll *= deg;
    }
};

