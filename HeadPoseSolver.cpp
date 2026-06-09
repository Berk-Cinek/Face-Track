#include "HeadPoseSolver.h"
#include <fstream>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

    HeadPoseSolver::HeadPoseSolver(int width, int height) : frame_width(width), frame_height(height)
    {
        // once it is wokring on cpu maybe look at getting it running with cuda aswell but it needs testing to get to that point just keep it in mind
        initLogger();
        initCameraMatrix();
        dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);
        mean_shape = loadMeanShape("meanshape_68.csv");
    }

    void HeadPoseSolver::solveAffine(const std::vector<cv::Point3d>& landmarks)
    {
        double s; //scale
        cv::Mat R, t; // R -> rotation matrix, t -> translation (cv::Mat, 3x1)

        cv::Mat landmarks_mat = cv::Mat(landmarks).reshape(1); //[68, 3]
        cv::Mat P = estimateAffine3Dto3D(mean_shape, landmarks_mat);
        P2sRt(P, s, R, t);
        spdlog::info("R values, (0,0): {:.1f} (0,1): {:.1f} (1,0)", R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(1, 0));
        matrix2angle(R, pitch, yaw, roll);
        spdlog::info("pitch: {:.1f}  yaw: {:.1f}  roll: {:.1f}", pitch, yaw, roll);
    }

    FaceData HeadPoseSolver::find_closest_face(const std::vector<FaceData>& faces)
    {
        cv::Point2d center(frame_width / 2, frame_height / 2);

        double bestDist = 1e18;
        FaceData best;

        for (auto& f : faces) {
            cv::Point2d c(
                f.bounding_box.x + f.bounding_box.width / 2,
                f.bounding_box.y + f.bounding_box.height / 2
            );
            double dx = c.x - center.x;
            double dy = c.y - center.y;
            double dist = dx * dx + dy * dy;

            if (dist < bestDist) {
                best = f;
                bestDist = dist;
            }
        }
        return best;
    }



    double HeadPoseSolver::get_yaw() {
        return yaw;
    };
    double HeadPoseSolver::get_pitch() {
        return pitch;
    };
    double HeadPoseSolver::get_roll() {
        return roll;
    };

    void HeadPoseSolver::initCameraMatrix()
    {
        double focal = frame_width * 1.2;
        cv::Point2d center(frame_width / 2, frame_height / 2);

        camera_matrix = (cv::Mat_<double>(3, 3) <<
            focal, 0, center.x,
            0, focal, center.y,
            0, 0, 1
            );

        spdlog::info("Camera matrix initialized.");
    }

    void HeadPoseSolver::initLogger()
    {
        try {
            auto cs = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            auto fs = std::make_shared<spdlog::sinks::basic_file_sink_mt>("headpose.log", true);
            std::vector<spdlog::sink_ptr> sinks{ cs, fs };

            auto logger = std::make_shared<spdlog::logger>("headpose_logger", sinks.begin(), sinks.end());
            spdlog::set_default_logger(logger);
            spdlog::set_level(spdlog::level::info);

            spdlog::info("Logger started.");
        }
        catch (...) {}
    }

    /*
    purely to visualize x,y,z axis on the nose might need some more work later to make it more
    "usefull" or just more accurate in general it feels like it barely moves
    also current code does not use it saving for future when i awnt to plot some visual elements

    void draw_pose_axis(cv::Mat& frame, const cv::Mat& rvec, const cv::Mat& tvec)
    {
        std::vector<cv::Point3d> axis = {
            {0,0,0},
            {100,0,0},
            {0,100,0},
            {0,0,100}
        };

        std::vector<cv::Point2d> image_points;
        cv::projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs, image_points);

        if (image_points.size() != 4)
            return;

        cv::Point p0 = image_points[0];
        cv::line(frame, p0, image_points[1], cv::Scalar(0, 0, 255), 2);
        cv::line(frame, p0, image_points[2], cv::Scalar(0, 255, 0), 2);
        cv::line(frame, p0, image_points[3], cv::Scalar(255, 0, 0), 2);
    }*/


    cv::Mat HeadPoseSolver::loadMeanShape(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            spdlog::error("loadMeanShape: could not open '{}'", path);
            throw std::runtime_error("loadMeanShape: failed to open " + path);
        }

        cv::Mat mean(68, 3, CV_64F);
        std::string line;
        int row = 0;
        while (std::getline(file, line) && row < 68)
        {
            std::stringstream ss(line);
            std::string val;
            int col = 0;
            while (std::getline(ss, val, ',') && col < 3) {
                mean.at<double>(row, col++) = std::stod(val);
            }
            row++;
        }
        return mean;
    }
    cv::Mat HeadPoseSolver::estimateAffine3Dto3D(const cv::Mat& X, const cv::Mat& Y) {
        int n = X.rows;
        cv::Mat ones = cv::Mat::ones(n, 1, CV_64F);
        cv::Mat X_homo;
        cv::hconcat(X, ones, X_homo);     //[68, 4]

        cv::Mat P_T;
        cv::solve(X_homo, Y, P_T, cv::DECOMP_SVD); //least squares
        return P_T.t();        //[3, 4]
    }

    void HeadPoseSolver::P2sRt(const cv::Mat& P, double& s, cv::Mat& R, cv::Mat& t) {
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
                                         
    void HeadPoseSolver::matrix2angle(const cv::Mat& R, double& pitch_matrix, double& yaw_matrix, double& roll_matrix) {
        for (int r = 0; r < 3; r++)
            spdlog::info("input matrix2angle R[{}]: {:.3f} {:.3f} {:.3f}", r, R.at<double>(r, 0), R.at<double>(r, 1), R.at<double>(r, 2));
        double sy = std::sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) +
            R.at<double>(1, 0) * R.at<double>(1, 0));

        if (sy > 1e-6) {
            pitch_matrix = std::atan2(R.at<double>(2, 1), R.at<double>(2, 2));
            yaw_matrix = std::atan2(-R.at<double>(2, 0), sy);
            roll_matrix = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0));
        }
        else {
            pitch_matrix = std::atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
            yaw_matrix = std::atan2(-R.at<double>(2, 0), sy);
            roll_matrix = 0.0;
        }
        for (int r = 0; r < 3; r++)
            spdlog::info("output matrix2angle R[{}]: {:.3f} {:.3f} {:.3f}", r, R.at<double>(r, 0), R.at<double>(r, 1), R.at<double>(r, 2));
        const double deg = 180.0 / CV_PI;
        pitch_matrix *= deg;
        yaw_matrix *= deg;
        roll_matrix *= deg;
    }
