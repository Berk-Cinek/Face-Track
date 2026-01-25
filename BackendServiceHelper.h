#pragma once
#include "BackendHelperModels.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <utility>

class BackEndServiceHelper
{
public:

    static inline float clampf(float v, float lo, float hi) {
        return std::max(lo, std::min(v, hi));
    }

    static inline cv::Point2d unletter_point(const cv::Point2d& pad, const letterBoxInfo& letterbox) {
        return cv::Point2d(
            (pad.x - letterbox.pad_x) / letterbox.scale,
            (pad.y - letterbox.pad_y) / letterbox.scale
        );
    }

    static inline cv::Rect2d unletter_rect(const cv::Rect2d& r, const letterBoxInfo& letterbox) {
        cv::Point2d p1 = unletter_point({ r.x, r.y }, letterbox);
        cv::Point2d p2 = unletter_point({ r.x + r.width, r.y + r.height }, letterbox);
        return cv::Rect2d(p1, p2);
    }

    static letterBoxInfo letterbox(const cv::Mat& src, cv::Mat& dst, int net_w, int net_h)
    {
        float scale = std::min(net_w / (float)src.cols, net_h / (float)src.rows);

        int new_w = int(src.cols * scale);
        int new_h = int(src.rows * scale);

        cv::Mat resized;
        cv::resize(src, resized, cv::Size(new_w, new_h));

        dst = cv::Mat::zeros(net_h, net_w, src.type());

        int pad_x = (net_w - new_w) / 2;
        int pad_y = (net_h - new_h) / 2;

        resized.copyTo(dst(cv::Rect(pad_x, pad_y, new_w, new_h)));

        return { scale, pad_x, pad_y, net_w, net_h };
    }

    static void fill_nchw_rgb(const cv::Mat& img, float* input, int H, int W)
    {
        const int HW = H * W;

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                cv::Vec3b bgr = img.at<cv::Vec3b>(y, x);

                float r = bgr[2] / 255.0f;
                float g = bgr[1] / 255.0f;
                float b = bgr[0] / 255.0f;

                input[0 * HW + y * W + x] = r;
                input[1 * HW + y * W + x] = g;
                input[2 * HW + y * W + x] = b;
            }
        }
    }

    static cv::Vec3d rvecToEulerDegrees(const cv::Mat& rvec)
    {
        cv::Mat R;
        cv::Rodrigues(rvec, R);

        double sy = std::sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) +
            R.at<double>(1, 0) * R.at<double>(1, 0));

        bool singular = sy < 1e-6;

        double x, y, z;
        if (!singular) {
            x = std::atan2(R.at<double>(2, 1), R.at<double>(2, 2)); // pitch
            y = std::atan2(-R.at<double>(2, 0), sy);               // yaw
            z = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0)); // roll
        }
        else {
            x = std::atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
            y = std::atan2(-R.at<double>(2, 0), sy);
            z = 0;
        }

        return {
            x * 180.0 / CV_PI,
            y * 180.0 / CV_PI,
            z * 180.0 / CV_PI
        };
    }

    static std::vector<FaceData> unletterbox_faces(const std::vector<FaceData>& faces640, const letterBoxInfo& lb, int orig_w, int orig_h) {
        std::vector<FaceData> out;
        out.reserve(faces640.size());

        for (const auto& f : faces640) {
            FaceData g = f;
            g.bounding_box = unletter_rect(f.bounding_box, lb);

            // clamp to original image bounds
            g.bounding_box.x = BackEndServiceHelper::clampf((float)g.bounding_box.x, 0.f, (float)orig_w - 1.f);
            g.bounding_box.y = BackEndServiceHelper::clampf((float)g.bounding_box.y, 0.f, (float)orig_h - 1.f);
            g.bounding_box.width = BackEndServiceHelper::clampf((float)g.bounding_box.width, 0.f, (float)orig_w - g.bounding_box.x);
            g.bounding_box.height = BackEndServiceHelper::clampf((float)g.bounding_box.height, 0.f, (float)orig_h - g.bounding_box.y);

            for (auto& lm : g.landmarks) {
                auto p = unletter_point(lm, lb);
                lm.x = BackEndServiceHelper::clampf((float)p.x, 0.f, (float)orig_w - 1.f);
                lm.y = BackEndServiceHelper::clampf((float)p.y, 0.f, (float)orig_h - 1.f);
            }
            out.push_back(std::move(g));
        }
        return out;
    }

};

