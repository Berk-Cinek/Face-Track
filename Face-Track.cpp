#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>
#include <filesystem>
#include <numeric> 
#include <chrono>

#define TICK(name)\
    auto t_##name = std::chrono::high_resolution_clock::now();

#define TOCK(name) \
    spdlog::info("{}: {} ms", #name, \
    std::chrono::duration_cast<std::chrono::milliseconds>( \
        std::chrono::high_resolution_clock::now() - t_##name).count());

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
    //taken only tvec.z for distance controll
    double tvec_z;
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

        const double threshold_rotation = 0.17; // rad
        const double threshold_distance = 0.30; // %

        cv::Mat R_raw, R_last;
        cv::Rodrigues(rvec_raw, R_raw);
        cv::Rodrigues(accepted_last.rvec, R_last);

        cv::Mat R_delta = R_raw * R_last.t();
        double trace = cv::trace(R_delta)[0];
        double cos_theta = (trace - 1.0) / 2.0;
        cos_theta = std::clamp(cos_theta, -1.0, 1.0);
        double rotation_angle = std::acos(cos_theta);

        double tz = tvec_raw.at<double>(2, 0);
        double tz_last = accepted_last.tvec_z;

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

        //3x1 matrix so to reach tvec.z you need to grab row 2 tvec = [tx] <- row 0 , [ty] <- row 1, [tz] <- row 2
        double tz = tvec_raw.at<double>(2, 0);

        //on the first run initilization of mainly rvec with .clone for a owned copy
        if (last_accepted.rvec.empty()) {
            last_accepted.rvec = rvec_raw.clone();
            last_accepted.tvec_z = tz;
        }

        cv::Mat R_raw;
        cv::Mat R_last;
        cv::Mat rvec_delta;
        cv::Mat R_step;
        
        
        cv::Rodrigues(rvec_raw, R_raw);//converting roations matricies to vectors
        cv::Rodrigues(last_accepted.rvec, R_last);

        cv::Mat R_delta = R_raw * R_last.t(); //.t() is the transpose --what rotation moves me from last -> now--
   
        cv::Rodrigues(R_delta, rvec_delta);

        double alpha_rot = 0.8;
        rvec_delta = rvec_delta * (1.0 - alpha_rot);//apply smoothing

        cv::Rodrigues(rvec_delta, R_step);
        cv::Mat R_new = R_step * R_last;
        cv::Rodrigues(R_new, last_accepted.rvec);

        last_accepted.tvec_z = 0.85 * last_accepted.tvec_z + (1 - 0.85) * tz;//linear smoothing

        spdlog::info("Tvec  after smoothing {:.1}", last_accepted.tvec_z);
        
    };

};

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

    letterBoxInfo letterbox(const cv::Mat& src, cv::Mat& dst, int net_w, int net_h)
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

    void fill_nchw_rgb(const cv::Mat& img, float* input, int H, int W)
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

    std::vector<FaceData> unletterbox_faces(const std::vector<FaceData>& faces640, const letterBoxInfo& lb, int orig_w, int orig_h) {
        std::vector<FaceData> out;
        out.reserve(faces640.size());

        for (const auto& f : faces640) {
            FaceData g = f;
            g.bounding_box = unletter_rect(f.bounding_box, lb);

            // clamp to original image bounds
            g.bounding_box.x = clampf((float)g.bounding_box.x, 0.f, (float)orig_w - 1.f);
            g.bounding_box.y = clampf((float)g.bounding_box.y, 0.f, (float)orig_h - 1.f);
            g.bounding_box.width = clampf((float)g.bounding_box.width, 0.f, (float)orig_w - g.bounding_box.x);
            g.bounding_box.height = clampf((float)g.bounding_box.height, 0.f, (float)orig_h - g.bounding_box.y);

            for (auto& lm : g.landmarks) {
                auto p = unletter_point(lm, lb);
                lm.x = clampf((float)p.x, 0.f, (float)orig_w - 1.f);
                lm.y = clampf((float)p.y, 0.f, (float)orig_h - 1.f);
            }
            out.push_back(std::move(g));
        }
        return out;
    }

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

            auto angles = rvecToEulerDegrees(rvec);

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

                    x1 = clampf(x1, 0.f, (float)img_w - 1.f);
                    y1 = clampf(y1, 0.f, (float)img_h - 1.f);
                    x2 = clampf(x2, 0.f, (float)img_w - 1.f);
                    y2 = clampf(y2, 0.f, (float)img_h - 1.f);

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
                FaceData f;
                f.bounding_box = all_boxes[idx];
                f.confidence = all_scores[idx];
                f.landmarks.reserve(5);
                for (auto& p : all_kps[idx]) f.landmarks.emplace_back(p.x, p.y);
                faces.push_back(std::move(f));
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


    int main()
    {
        static FacePoseController controller;

        // Ort model config
        std::int64_t batch = 1;
        std::int64_t numchannels = 3;
        std::int64_t width = 640;
        std::int64_t height = 640;
        std::vector <int64_t> input_shape = { batch, numchannels, height, width };
        size_t input_tensor_size = numchannels * height * width;

        std::int64_t numInputElements = batch * numchannels * height * width;

        // Ort setup
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "scrfd");

        Ort::SessionOptions session_options;
        //by not setting this option ort autmaticly manages cores and affinitisez them
        //session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        Ort::Session session(env, L"scrfd_model.onnx", session_options);


        std::vector <float> input_buffer(input_tensor_size);
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);//check later if OrtMemType... will need to be cuda specific

        // input/output names
        Ort::AllocatorWithDefaultOptions allocator;

        auto input_name = session.GetInputNameAllocated(0, allocator);
        std::vector<const char*> input_names = { input_name.get() };

        auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
        std::vector<const char*> output_names;
        std::vector<Ort::AllocatedStringPtr> output_name_allocs;

        for (size_t i = 0; i < session.GetOutputCount(); ++i) {
            output_name_allocs.push_back(session.GetOutputNameAllocated(i, allocator));
            output_names.push_back(output_name_allocs.back().get());
        }

        HeadPoseSolver solver(640, 480);

        //openCv camera
        cv::VideoCapture cap(0);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

        if (!cap.isOpened()) {
            spdlog::error("Cannot open camera!");
            return -1;
        }

        while (true)
        {
            cv::Mat frame;
            cap >> frame;
            if (frame.empty())
                break;

            cv::Mat img640;
            letterBoxInfo lb = letterbox(frame, img640, 640, 640);

            fill_nchw_rgb(img640, input_buffer.data(), 640, 640);

            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                input_buffer.data(),
                input_buffer.size(),
                input_shape.data(),
                input_shape.size()
            );

            try {
                auto outputs = session.Run(
                    Ort::RunOptions{ nullptr },
                    input_names.data(),
                    &input_tensor,
                    1,
                    output_names.data(),
                    output_names.size()
                );

                auto faces640 = solver.parse_scrfd_ort(outputs, 640, 640);
                auto faces = unletterbox_faces(faces640, lb, frame.cols, frame.rows);


                if (!faces.empty()) {
                    //find closest face and ALWAYS use that
                    FaceData target = solver.find_closest_face(faces);

                    solver.solve(frame, target);

                    if (controller.pose_valid == false) {
                        controller.initialize(faces, solver.get_rvec(), solver.get_tvec());
                    }
                    else {
                        controller.update(faces, solver.get_rvec(), solver.get_tvec());
                        solver.angelDistanceFind();
                        solver.angelDistanceDraw(frame, controller.last_face);
                    }
                }


            }
            catch (const Ort::Exception& e) {
                std::cerr << "ORT ERROR: " << e.what() << std::endl;
                return -1;
            }

            cv::imshow("Camera", frame);
            if (cv::waitKey(1) == 27)
                break;
        }

        return 0;
    }