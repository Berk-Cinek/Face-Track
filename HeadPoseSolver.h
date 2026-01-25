#pragma once
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <onnxruntime_cxx_api.h>
#include "BackEndServiceHelper.h"


class HeadPoseSolver {
public:
    HeadPoseSolver(int width, int height) : frame_width(width), frame_height(height)
    {
        // once it is wokring on cpu maybe look at getting it running with cuda aswell but it needs testing to get to that point just keep it in mind
        initLogger();
        initCameraMatrix();
        dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);
    }

    //solvePNP function
    void solve(cv::Mat& /*frame*/, const FaceData& face) {
        if (face.landmarks.size() != model_points.size()) {
            return;
        }

        // Warm start after first solve
        bool use_guess = pose_initialized;

        std::vector<int> inliers;
        bool ok = cv::solvePnPRansac(
            model_points,
            face.landmarks,
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
            use_guess,
            100,         // iterations
            3.0,         // reprojection error
            0.99,        // confidence
            inliers,
            cv::SOLVEPNP_ITERATIVE
        );

        if (!ok) return;

        try {
            cv::solvePnPRefineLM(
                model_points,
                face.landmarks,
                camera_matrix,
                dist_coeffs,
                rvec,
                tvec
            );
        }
        catch (...) {
        }

        pose_initialized = true;
    }

    void angelDistanceFind() {

        double distance_mm = tvec.at<double>(2);
        distance_cm = distance_mm / 10.0;

        auto angles = BackEndServiceHelper::rvecToEulerDegrees(rvec);

        pitch = angles[0];
        yaw = angles[1];
        roll = angles[2];
    }

    void angelDistanceDraw(cv::Mat& frame, const FaceData& face) {

        char buf[64];
        char text[128];
        spdlog::info("Distance to camera: {:.1f} cm", distance_cm);

        cv::rectangle(frame, face.bounding_box, cv::Scalar(0, 255, 0), 2);

        draw_pose_axis(frame, rvec, tvec);
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
        return rvec;
    }
    cv::Mat get_tvec() {
        return tvec;
    }

private:
    cv::dnn::Net face_detector;
    int frame_width, frame_height;
    cv::Mat camera_matrix, dist_coeffs, rvec, tvec;
    double pitch, yaw, roll, distance_cm;
    bool pose_initialized = false;

    std::vector<cv::Point3d> model_points{
        {-32.0,  32.0, -30.0},  // left eye center
        { 32.0,  32.0, -30.0},  // right eye center
        {  0.0,   0.0,   0.0},  // nose tip
        {-22.0, -28.0, -20.0},  // left mouth corner
        { 22.0, -28.0, -20.0}   // right mouth corner
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
};
