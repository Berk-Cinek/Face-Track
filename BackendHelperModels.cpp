#include "BackendHelperModels.h"
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector> 


    void PoseKalmanFilter::initChannel(cv::KalmanFilter& kf) {
        kf.init(2, 1, 0, CV_32F);
        kf.transitionMatrix = (cv::Mat_<float>(2, 2) << 1, 1, 0, 1);
        kf.measurementMatrix = (cv::Mat_<float>(1, 2) << 1, 0);

        cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-4)); //Q tuning
        cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-2)); //R tuning
        cv::setIdentity(kf.errorCovPost, cv::Scalar(1.0));
    }

    float PoseKalmanFilter::updateChannel(cv::KalmanFilter& kf, float measurement)
    {
        kf.predict();
        cv::Mat measure = (cv::Mat_<float>(1, 1) << measurement);
        cv::Mat corrected = kf.correct(measure);
        return corrected.at<float>(0);
    }

    PoseKalmanFilter::PoseKalmanFilter() {
        for (auto& kf : channels)
            initChannel(kf);
    }

    void PoseKalmanFilter::filter(const cv::Mat& rvec_in, const cv::Mat& tvec_in, cv::Mat& rvec_out, cv::Mat& tvec_out) {
        double rv[3], tv[3];
        for (int i = 0; i < 3; i++)
        {
            rv[i] = updateChannel(channels[i], (float)rvec_in.at<double>(i));
            tv[i] = updateChannel(channels[i + 3], (float)tvec_in.at<double>(i));
        }
        rvec_out = (cv::Mat_<double>(3, 1) << rv[0], rv[1], rv[2]);
        tvec_out = (cv::Mat_<double>(3, 1) << tv[0], tv[1], tv[2]);
    }

    void FacePoseController::initialize(const std::vector<FaceData>& faces, cv::Mat rvec, cv::Mat tvec) {
        if (!pose_valid && !faces.empty()) {
            last_accepted.rvec = rvec.clone();
            last_accepted.tvec = tvec.clone();
            last_face = faces[0];
            has_face = true;
            lost_frames = 0;
            pose_valid = true;
        }
    }

    void FacePoseController::update(const std::vector<FaceData>& faces, cv::Mat rvec_raw, cv::Mat tvec_raw) {

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

