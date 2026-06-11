#include "Parser.h"
#include <spdlog/spdlog.h>

std::vector<FaceData> Parser::parse_scrfd_ort(const std::vector<Ort::Value>& outputs, int img_w, int img_h, float conf_thresh, float nms_thresh) {
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

            x1 = std::clamp(x1, 0.f, (float)img_w - 1.f);
            y1 = std::clamp(y1, 0.f, (float)img_h - 1.f);
            x2 = std::clamp(x2, 0.f, (float)img_w - 1.f);
            y2 = std::clamp(y2, 0.f, (float)img_h - 1.f);

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

std::vector<cv::Point3d> Parser::parse_1k3d68_ort(const std::vector<Ort::Value>& outputs, const cv::Mat& M) {

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