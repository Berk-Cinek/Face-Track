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

struct FaceData {
    cv::Rect2d bounding_box;
    std::vector<cv::Point2d> landmarks;
    float confidence = 0.0f;
};

class HeadPoseSolver {
public:
    HeadPoseSolver(int width, int height): frame_width(width), frame_height(height)
    {

        // CUDA is gonna disabled for now come back to this later as getting errors that cuda is not enabled fully or something
        //face_detector.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        //face_detector.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

        initLogger();
        initCameraMatrix();
        dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);
    }

    void solveAndDraw(cv::Mat& frame, const FaceData& face) {
        if (face.landmarks.size() != model_points.size())
            return;

        cv::Mat rvec, tvec;

        cv::solvePnP(
            model_points,
            face.landmarks,
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
            cv::SOLVEPNP_EPNP
        );

        cv::rectangle(frame, face.bounding_box, cv::Scalar(0, 255, 0), 2);
        draw_pose_axis(frame, rvec, tvec);
    }


    std::vector<FaceData> parse_scrfd_output(const std::vector<cv::Mat>& output_blobs, int64_t img_w, int64_t img_h)
    {
        const float CONF_THRESH = 0.5f;
        const float NMS_THRESH = 0.45f;
        //safety check 9 blobs expected
        if (output_blobs.size() < 9) {
            spdlog::error("SCRFD: expected 9 output blobs, got {}", output_blobs.size());
            return{};
        }

        const int strides[3] = { 8, 16, 32 };
        std::vector<cv::Rect> all_boxes;
        std::vector<float> all_scores;
        std::vector<std::vector<cv::Point2f>> all_landmarks;


        for (int level = 0; level < 3; ++level) {
            int stride = strides[level];

            const cv::Mat& bbox_blob = output_blobs[level * 3 + 0];
            const cv::Mat& cls_blob = output_blobs[level * 3 + 1];
            const cv::Mat& kps_blob = output_blobs[level * 3 + 2];

            CV_Assert(bbox_blob.dims == 4 && cls_blob.dims == 4 && kps_blob.dims == 4);

            int H = cls_blob.size[2];
            int W = cls_blob.size[3];

            //channels
            int C_cls = cls_blob.size[1];
            int C_bbox = bbox_blob.size[1];
            int C_kps = kps_blob.size[1];

            //anchors per cell (A)
            int A = C_cls / 2;

            CV_Assert(C_bbox == A * 4);
            CV_Assert(C_kps == A * 10);

            //raw data pointers
            const float* cls_data = (const float*)cls_blob.data;
            const float* bbox_data = (const float*)bbox_blob.data;
            const float* kps_data = (const float*)kps_blob.data;

            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    for (int a = 0; a < A; ++a) {

                        //class index for "face"
                        int c_face = a * 2 + 1;

                        //memory index helper for NCHW layout
                        int cls_offset = ((c_face * H) + h) * W + w;
                        float score = cls_data[cls_offset];

                        if (score < CONF_THRESH)
                            continue;

                        float cx = (w + 0.5f) * stride;
                        float cy = (h + 0.5f) * stride;

                        // 4 bbox channels for this anchor
                        float dx = bbox_data[((a * 4 + 0) * H + h) * W + w] * stride;
                        float dy = bbox_data[((a * 4 + 1) * H + h) * W + w] * stride;
                        float dw = bbox_data[((a * 4 + 2) * H + h) * W + w] * stride;
                        float dh = bbox_data[((a * 4 + 3) * H + h) * W + w] * stride;

                        float x1 = cx - dx;
                        float y1 = cy - dy;
                        float x2 = cx + dw;
                        float y2 = cy + dh;

                        x1 = std::max(0.0f, std::min(x1, (float)img_w - 1));
                        y1 = std::max(0.0f, std::min(y1, (float)img_h - 1));
                        x2 = std::max(0.0f, std::min(x2, (float)img_w - 1));
                        y2 = std::max(0.0f, std::min(y2, (float)img_h - 1));

                        int box_w = int(x2 - x1 + 0.5f);
                        int box_h = int(y2 - y1 + 0.5f);
                        if (box_w <= 0 || box_h <= 0)
                            continue;

                        cv::Rect box((int)x1, (int)y1, box_w, box_h);

                        std::vector<cv::Point2f> landmarks;
                        landmarks.reserve(5);
                        for (int j = 0; j < 5; ++j) {
                            int base = a * 10 + j * 2;
                            float lx = kps_data[((base + 0) * H + h) * W + w] * stride + cx;
                            float ly = kps_data[((base + 1) * H + h) * W + w] * stride + cy;
                            landmarks.emplace_back(lx, ly);
                        }

                        all_boxes.push_back(box);
                        all_scores.push_back(score);
                        all_landmarks.push_back(std::move(landmarks));
                    }
                }
            }
        }

        //finally, run NMS on all
        std::vector<int> keep;
        cv::dnn::NMSBoxes(all_boxes, all_scores, CONF_THRESH, NMS_THRESH, keep);

        std::vector<FaceData> faces;
        faces.reserve(keep.size());
        for (int idx : keep) {
            FaceData f;
            f.bounding_box = all_boxes[idx];
            f.confidence = all_scores[idx];
            for (auto& p : all_landmarks[idx]) {
                f.landmarks.emplace_back(p.x, p.y);
            }
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

private:
    cv::dnn::Net face_detector;
    int frame_width, frame_height;
    cv::Mat camera_matrix, dist_coeffs;

    std::vector<cv::Point3d> model_points{
        {0,0,0},
        {225,170,-135},
        {-225,170,-135},
        {150,-150,-125},
        {-150,-150,-125}
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

void opencvConverstion(cv::Mat& frame, int64_t height, int64_t width, float* input_buffer) {

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(640, 640));
    int64_t heightWidth = height * width;

    for (int64_t y = 0; y < height; ++y) {
        for (int64_t x = 0; x < width; ++x) {

            cv::Vec3b pix = resized.at<cv::Vec3b>(y, x);

            float blue = pix[0] / 255.0f;
            float green = pix[1] / 255.0f;
            float red = pix[2] / 255.0f;

            input_buffer[0 * heightWidth + y * width + x] = red;
            input_buffer[1 * heightWidth + y * width + x] = green;
            input_buffer[2 * heightWidth + y * width + x] = blue;
        };
    };
};

int main()
{
    // Ort model config
    std::int64_t batch = 1;
    std::int64_t numchannels = 3;
    std::int64_t width = 640;
    std::int64_t height = 640;
    std::vector <int64_t> input_shape = { batch, numchannels, height, width };
    size_t input_tensor_size = numchannels * height * width;

    std::vector <float> input_buffer(numchannels * height * width);

    std::int64_t numInputElements = batch * numchannels * height * width;

    // Ort setup
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "scrfd");

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, L"scrfd_model.onnx", session_options);
    
    std::vector <float> input_buffer(input_tensor_size); 
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);//check later if OrtMemType... will need to be cuda specific

    // input/output names
    Ort::AllocatorWithDefaultOptions allocator;

    auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
    const char* input_name = input_name_alloc.get();
    std::vector <const char*> input_names{ input_name };

    std::vector <const char*> output_names;
    for (size_t i = 0; i < session.GetOutputCount(); ++i) {
        output_names.push_back(session.GetInputNameAllocated(i, allocator).get());
    };

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
        
        opencvConverstion(frame, height, width, input_buffer.data());

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_buffer.data(),
            input_buffer.size(),
            input_shape.data(),
            input_shape.size()
            );

        //inference
        std::vector<Ort::Value> outputs = session.Run(
            Ort::RunOptions{ nullptr },
            input_names.data(),
            &input_tensor,
            1,
            output_names.data(),
            output_names.size()
        );

        // --- TODO ---
       // parse_scrfd_output(outputs, frame.cols, frame.rows);

        //pick face + solve pose
        //if(!face.empthy()) solver.solveAndDraw(frame, faces[0]);

        cv::imshow("Camera", frame);
        if (cv::waitKey(1) == 27)
            break;
    }

    return 0;
}

