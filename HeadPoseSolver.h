#pragma once
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <onnxruntime_cxx_api.h>
#include <fstream>
#include "BackEndServiceHelper.h"
#include "BackendHelperModels.h"


class HeadPoseSolver {
public:
    HeadPoseSolver(int width, int height) : frame_width(width), frame_height(height)
    {
        // once it is wokring on cpu maybe look at getting it running with cuda aswell but it needs testing to get to that point just keep it in mind
        initLogger();
        initCameraMatrix();
        dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);
        mean_shape = loadMeanShape("meanshape_68.csv");
    }

    void solveAffine(const std::vector<cv::Point3d>& landmarks)
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


    void angelDistanceFind(const cv::Mat& smooth_rvec, const cv::Mat& smooth_tvec) {

        double distance_mm = smooth_tvec.at<double>(2);
        distance_cm = distance_mm / 10.0;

        auto angles = BackEndServiceHelper::rvecToEulerDegrees(smooth_rvec);

        pitch = angles[0];
        yaw = angles[1];
        roll = angles[2];
    }

    void angelDistanceDraw(cv::Mat& frame, const FaceData& face, const cv::Mat& smooth_rvec, const cv::Mat& smooth_tvec) {

        char buf[64];
        char text[128];
        spdlog::info("Distance to camera: {:.1f} cm", distance_cm);

        cv::rectangle(frame, face.bounding_box, cv::Scalar(0, 255, 0), 2);

        draw_pose_axis(frame, smooth_rvec, smooth_tvec);
        std::snprintf(
            text, sizeof(text),
            "Pitch: %5.1f  Yaw: %5.1f  Roll: %5.1f",
            pitch, yaw, roll
        );

        // yaw pitch Display
        cv::putText(
            frame,
            text,
            cv::Point(20, 30),
            cv::FONT_HERSHEY_SIMPLEX,
            0.7,
            cv::Scalar(0, 255, 255),
            2,
            cv::LINE_AA
        );

        //distance display
        std::snprintf(buf, sizeof(buf), "Dist: %.1f cm", distance_cm);

        cv::putText(
            frame,
            buf,
            { 20, 80 },
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            { 0, 255, 255 },
            2);


        spdlog::info(
            "Pitch: {:.1f} degrees, Yaw: {:.1f} degrees, Roll: {:.1f} degrees",
            pitch, yaw, roll
        );
    }

    std::vector<FaceData> parse_scrfd_ort(const std::vector<Ort::Value>& outputs, int img_w, int img_h, float conf_thresh = 0.5f, float nms_thresh = 0.45f) {
        std::vector<FaceData> faces;

        if (outputs.size() != 9) {
            spdlog::error("SCRFD: expected 9 outputs, got {}", outputs.size());
            return faces;
        }

        //printed order:
        // score_8, 16, 32, bbox_8, 16, 32, kps_8, 16, 32
        // indices are:
        // scores: 0..2
        // bbox:   3..5
        // kps:    6..8
        const int strides[3] = { 8, 16, 32 };

        std::vector<cv::Rect> all_boxes;
        std::vector<float> all_scores;
        std::vector<std::vector<cv::Point2f>> all_kps;

        for (int level = 0; level < 3; ++level) {
            const int stride = strides[level];

            const Ort::Value& score_t = outputs[level + 0];
            const Ort::Value& bbox_t = outputs[level + 3];
            const Ort::Value& kps_t = outputs[level + 6];

            // shape checks
            auto s_shape = score_t.GetTensorTypeAndShapeInfo().GetShape();
            auto b_shape = bbox_t.GetTensorTypeAndShapeInfo().GetShape();
            auto k_shape = kps_t.GetTensorTypeAndShapeInfo().GetShape();

            if (s_shape.size() != 2 || b_shape.size() != 2 || k_shape.size() != 2) {
                spdlog::error("SCRFD: unexpected tensor rank at level {}", level);
                continue;
            }

            const int64_t N = s_shape[0];
            if (s_shape[1] != 1 || b_shape[0] != N || b_shape[1] != 4 || k_shape[0] != N || k_shape[1] != 10) {
                spdlog::error("SCRFD: unexpected shapes at level {}: score[{},{}], bbox[{},{}], kps[{},{}]",
                    level,
                    (long long)s_shape[0], (long long)s_shape[1],
                    (long long)b_shape[0], (long long)b_shape[1],
                    (long long)k_shape[0], (long long)k_shape[1]);
                continue;
            }

            const float* score = score_t.GetTensorData<float>();
            const float* bbox = bbox_t.GetTensorData<float>();
            const float* kps = kps_t.GetTensorData<float>();

            const int Wg = img_w / stride;
            const int Hg = img_h / stride;

            const int cells = Hg * Wg;
            if (cells <= 0 || (N % cells) != 0) {
                spdlog::error("SCRFD: cannot infer anchors/cell at stride {} (N={}, cells={})", stride, (long long)N, cells);
                continue;
            }
            const int A = (int)(N / cells);

            for (int i = 0; i < (int)N; ++i) {
                float sc = score[i];
                if (sc < conf_thresh) continue;

                int cell_index = i / A;
                int a = i % A;

                int gy = cell_index / Wg;
                int gx = cell_index % Wg;

                float cx = (gx + 0.5f) * stride;
                float cy = (gy + 0.5f) * stride;

                float dx = bbox[i * 4 + 0] * stride;
                float dy = bbox[i * 4 + 1] * stride;
                float dw = bbox[i * 4 + 2] * stride;
                float dh = bbox[i * 4 + 3] * stride;

                float x1 = cx - dx;
                float y1 = cy - dy;
                float x2 = cx + dw;
                float y2 = cy + dh;

                x1 = BackEndServiceHelper::clampf(x1, 0.f, (float)img_w - 1.f);
                y1 = BackEndServiceHelper::clampf(y1, 0.f, (float)img_h - 1.f);
                x2 = BackEndServiceHelper::clampf(x2, 0.f, (float)img_w - 1.f);
                y2 = BackEndServiceHelper::clampf(y2, 0.f, (float)img_h - 1.f);

                int bw = (int)(x2 - x1 + 0.5f);
                int bh = (int)(y2 - y1 + 0.5f);
                if (bw <= 0 || bh <= 0) continue;

                cv::Rect box((int)x1, (int)y1, bw, bh);

                std::vector<cv::Point2f> lm;
                lm.reserve(5);
                for (int j = 0; j < 5; ++j) {
                    float lx = kps[i * 10 + (j * 2 + 0)] * stride + cx;
                    float ly = kps[i * 10 + (j * 2 + 1)] * stride + cy;
                    lm.emplace_back(lx, ly);
                }

                all_boxes.push_back(box);
                all_scores.push_back(sc);
                all_kps.push_back(std::move(lm));
            }
        }

        if (all_boxes.empty()) return faces;

        std::vector<int> keep;
        cv::dnn::NMSBoxes(all_boxes, all_scores, conf_thresh, nms_thresh, keep);

        faces.reserve(keep.size());
        for (int idx : keep) {
            FaceData facedata;
            facedata.bounding_box = all_boxes[idx];
            facedata.confidence = all_scores[idx];
            facedata.landmarks.reserve(5);
            for (auto& p : all_kps[idx]) facedata.landmarks.emplace_back(p.x, p.y);
            faces.push_back(std::move(facedata));
        }
        return faces;
    }

    std::vector<cv::Point3d> parse_1k3d68_ort(const std::vector<Ort::Value>& outputs, const cv::Mat& M) {
        
        const float* data = outputs[0].GetTensorData<float>();
        std::vector<cv::Point3d> landmarks;
        landmarks.reserve(68);
        cv::Mat M_inv;
        cv::invertAffineTransform(M, M_inv);
        double scale = std::sqrt(M_inv.at<double>(0, 0) * M_inv.at<double>(0, 0) + M_inv.at<double>(1, 0) * M_inv.at<double>(1, 0));

        for (int i = 1035; i < 1103; i++)
        {
            float raw_x = data[i * 3 + 0];
            float raw_y = data[i * 3 + 1];
            float raw_z = data[i * 3 + 2];

            float x_crop = (raw_x + 1.0f) * 96.0f;
            float y_crop = (raw_y + 1.0f) * 96.0f;
            float z_crop = raw_z * 96.0f;

            double fx = M_inv.at<double>(0, 0) * x_crop + M_inv.at<double>(0, 1) * y_crop + M_inv.at<double>(0, 2);
            double fy = M_inv.at<double>(1, 0) * x_crop + M_inv.at<double>(1, 1) * y_crop + M_inv.at<double>(1, 2);

            //scaling z as M_inv is not a 3D vector
            double fz = z_crop * scale;
            spdlog::info("nose raw_z: {:.4f}  fz: {:.2f}  fx: {:.2f}  fy: {:.2f}", raw_z, fz, fx, fy);

            landmarks.emplace_back(fx, fy, fz);
        }
        spdlog::info("Nose tip x: {:.1f}, y: {:.1f}", landmarks[30].x, landmarks[30].y);
        return landmarks;
    }

    FaceData find_closest_face(const std::vector<FaceData>& faces)
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

    

    double get_yaw() {
        return yaw;
    };
    double get_pitch() {
        return pitch;
    };
    double get_roll() {
        return roll;
    };
    double get_distance() {
        return distance_cm;
    };
    cv::Mat get_rvec() {
        return rvec_smooth;
    }
    cv::Mat get_tvec() {
        return tvec_smooth;
    }

private:
    cv::dnn::Net face_detector;
    int frame_width, frame_height;
    cv::Mat camera_matrix, dist_coeffs, rvec, tvec;
    double pitch, yaw, roll, distance_cm;
    bool pose_initialized = false;
    PoseKalmanFilter kalaman;
    cv::Mat rvec_smooth, tvec_smooth;
    cv::Mat mean_shape;

    std::vector<cv::Point3d> model_points{
      { -32.0, -27.0, -20.0 },  // left eye
      {  32.0, -27.0, -20.0 },  // right eye
      {   0.0,   0.0,   0.0 },  // nose tip
      { -27.0,  27.0, -22.0 },  // left mouth
      {  27.0,  27.0, -22.0 }   // right mouth2
    };

    void initCameraMatrix()
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

    void initLogger()
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

    //purely to visualize x,y,z axis on the nose might need some more work later to make it more "usefull" or just more accurate in general it feels like it barely moves
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
    }

    cv::Mat loadMeanShape(const std::string& path) {
        cv::Mat mean(68, 3, CV_64F);
        std::ifstream file(path);
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
        mean.col(0) *= -1.0;
        return mean;
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
    void matrix2angle(const cv::Mat& R, double& pitch_matrix, double& yaw_matrix, double& roll_matrix) {
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
};
