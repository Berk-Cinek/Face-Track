#include "BackEndServiceHelper.h"
#include "BackendHelperModels.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>             
#include <utility>  

    

    cv::Point2d LetterBoxGeometry::unletter_point(const cv::Point2d& pad, const letterBoxInfo& letterbox) {
        return cv::Point2d(
            (pad.x - letterbox.pad_x) / letterbox.scale,
            (pad.y - letterbox.pad_y) / letterbox.scale
        );
    }

    cv::Rect2d LetterBoxGeometry::unletter_rect(const cv::Rect2d& r, const letterBoxInfo& letterbox) {
        cv::Point2d p1 = LetterBoxGeometry::unletter_point({ r.x, r.y }, letterbox);
        cv::Point2d p2 = LetterBoxGeometry::unletter_point({ r.x + r.width, r.y + r.height }, letterbox);
        return cv::Rect2d(p1, p2);
    }

    letterBoxInfo LetterBoxGeometry::letterbox(const cv::Mat& src, cv::Mat& dst, int net_w, int net_h)
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

    std::vector<FaceData> LetterBoxGeometry::unletterbox_faces(const std::vector<FaceData>& faces640, const letterBoxInfo& lb, int orig_w, int orig_h) {
        std::vector<FaceData> out;
        out.reserve(faces640.size());

        for (const auto& f : faces640) {
            FaceData g = f;
            g.bounding_box = unletter_rect(f.bounding_box, lb);

            // clamp to original image bounds
            g.bounding_box.x = std::clamp(g.bounding_box.x, 0.0, (double)orig_w - 1.0);
            g.bounding_box.y = std::clamp(g.bounding_box.y, 0.0, (double)orig_h - 1.0);
            g.bounding_box.width = std::clamp(g.bounding_box.width, 0.0, (double)orig_w - g.bounding_box.x);
            g.bounding_box.height = std::clamp(g.bounding_box.height, 0.0, (double)orig_h - g.bounding_box.y);

            for (auto& lm : g.landmarks) {
                auto p = unletter_point(lm, lb);
                lm.x = std::clamp((float)p.x, 0.f, (float)orig_w - 1.f);
                lm.y = std::clamp((float)p.y, 0.f, (float)orig_h - 1.f);
            }
            out.push_back(std::move(g));
        }
        return out;
    }

    void ModelInput::pack_nchw_rgb(const cv::Mat& img, float* out, float mean, float scale) {
        const int H = img.rows, W = img.cols, HW = H * W;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const cv::Vec3b bgr = img.at<cv::Vec3b>(y, x);
                out[0 * HW + y * W + x] = (bgr[2] - mean) * scale; // R
                out[1 * HW + y * W + x] = (bgr[1] - mean) * scale; // G
                out[2 * HW + y * W + x] = (bgr[0] - mean) * scale; // B
            }
    }
    

    cv::Mat ModelInput::cropFaceFor1k3d68(const cv::Mat& frame, const FaceData& face, cv::Mat& dst, float* input_buffer)
    {
        //source points
        cv::Point2f src[3];
        src[0] = cv::Point2f(face.bounding_box.x, face.bounding_box.y);
        src[1] = cv::Point2f(face.bounding_box.x + face.bounding_box.width, face.bounding_box.y);
        src[2] = cv::Point2f(face.bounding_box.x, face.bounding_box.y + face.bounding_box.height);

        //destination points
        cv::Point2f dest[3];
        dest[0] = cv::Point2f(0, 0);
        dest[1] = cv::Point2f(192, 0);
        dest[2] = cv::Point2f(0, 192);


        //streching is acceptable because The affine transform M records exactly how the image was stretched,
        //so when you map the landmarks back using the inverse of M, they land in the correct position in the original frame
        cv::Mat M = cv::getAffineTransform(src, dest);
        cv::warpAffine(frame, dst, M, cv::Size(192, 192));

        //fill nchw_rgb for the newly cropped image
        ModelInput::pack_nchw_rgb(dst, input_buffer, 127.5f, 1.0f / 128.0f);

        return M;
    }


