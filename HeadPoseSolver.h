#pragma once
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include "BackendHelperModels.h"
#include "BackEndServiceHelper.h"


class HeadPoseSolver {
public:
    HeadPoseSolver(int width, int height);

    void solveAffine(const std::vector<cv::Point3d>& landmarks);

    FaceData find_closest_face(const std::vector<FaceData>& faces);

    double get_yaw();
    double get_pitch();
    double get_roll();

private:
    cv::dnn::Net face_detector;
    int frame_width, frame_height;
    cv::Mat camera_matrix, dist_coeffs;
    double pitch, yaw, roll, distance_cm;
    cv::Mat rvec_smooth, tvec_smooth;
    cv::Mat mean_shape;

    void initCameraMatrix();

    void initLogger();

    cv::Mat loadMeanShape(const std::string& path);

    // estimate 3x4 affine matrix mapping mean shape X onto predicted landmarks Y
    cv::Mat estimateAffine3Dto3D(const cv::Mat& X, const cv::Mat& Y);

    // decompose affine matrix into scale, rotation, translation
    void P2sRt(const cv::Mat& P, double& s, cv::Mat& R, cv::Mat& t);

    // rotation matrix → pitch, yaw, roll in degrees                                         
    void matrix2angle(const cv::Mat& R, double& pitch_matrix, double& yaw_matrix, double& roll_matrix);
};
