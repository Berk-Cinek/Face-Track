#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <vector>
#include <iostream>
#include <numeric>

struct FaceData {
    cv::Rect2d bounding_box;
    std::vector<cv::Point2d> landmarks;
    float confidence;
};

class HeadPoseEstimator {
public:
    HeadPoseEstimator(const std::string& scrfd_model_path, int width, int height)
        : frame_width(width), frame_height(height)
    {
        face_detector = cv::dnn::readNetFromONNX(scrfd_model_path);

        // CUDA stays enabled!
        face_detector.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        face_detector.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

        initLogger();
        initCameraMatrix();
        dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);
    }

    std::vector<cv::Mat> convertOutput(const std::vector<cv::Mat>& output_blobs)
    {
        std::vector<cv::Mat> result;
        result.reserve(output_blobs.size());

        for (const cv::Mat& blob : output_blobs)
        {
            if (blob.empty()) {
                result.push_back(cv::Mat());
                continue;
            }

            cv::Mat flat = blob.clone();
            flat = flat.reshape(1, 1);  // flatten to 1 row
            result.push_back(flat);
        }
        return result;
    }
    void process_frame(cv::Mat& frame)
    {
        cv::Mat blob_cpu;
        cv::dnn::blobFromImage(
            frame, blob_cpu,
            1.0,
            cv::Size(640, 480),
            cv::Scalar(104, 117, 123),
            false, false
        );
        face_detector.setInput(blob_cpu);

        std::vector<cv::Mat> raw_outputs;
        try {
            face_detector.forward(raw_outputs, face_detector.getUnconnectedOutLayersNames());
        }
        catch (const cv::Exception& e) {
            std::cout << "🔥 DNN ERROR: " << e.what() << std::endl;
            return;
        }
        auto cpu_outputs = convertOutput(raw_outputs);
        auto faces = parse_scrfd_output(cpu_outputs, frame.cols, frame.rows);
        if (faces.empty())
            return;

        FaceData main_face = find_closest_face(faces);

        if (main_face.landmarks.size() != model_points.size())
            return;

        cv::Mat rvec, tvec;

        try {
            cv::solvePnP(
                model_points, main_face.landmarks,
                camera_matrix, dist_coeffs,
                rvec, tvec,
                false, cv::SOLVEPNP_EPNP
            );
        }
        catch (const cv::Exception& e) {
            std::cout << "🔥 PNP ERROR: " << e.what() << std::endl;
            return;
        }

        cv::rectangle(frame, main_face.bounding_box, cv::Scalar(0, 255, 0), 2);
        draw_pose_axis(frame, rvec, tvec);
    }
    //needs new parse logic incoming too big/not the right size
    std::vector<FaceData> parse_scrfd_output(const std::vector<cv::Mat>& output_blobs, int img_w, int img_h)
    {
        const float CONFIDENCE_THRESHOLD = 0.5f;
        const float NMS_THRESHOLD = 0.45f;

        if (output_blobs.size() < 3)
            return {};

        const cv::Mat& boxes_blob = output_blobs[0];
        const cv::Mat& scores_blob = output_blobs[1];
        const cv::Mat& landmarks_blob = output_blobs[2];

        if (boxes_blob.empty() || scores_blob.empty() || landmarks_blob.empty())
            return {};

        std::vector<cv::Rect> bboxes;
        std::vector<float> confidences;
        std::vector<std::vector<cv::Point2f>> all_landmarks_temp;

        int num_predictions = scores_blob.cols / 2;

        const float* scores_data = (float*)scores_blob.data;
        const float* boxes_data = (float*)boxes_blob.data;
        const float* lm_data = (float*)landmarks_blob.data;

        for (int i = 0; i < num_predictions; i++)
        {
            float conf = scores_data[i * 2 + 1];
            if (conf < CONFIDENCE_THRESHOLD)
                continue;

            float x1 = boxes_data[i * 4 + 0];
            float y1 = boxes_data[i * 4 + 1];
            float x2 = boxes_data[i * 4 + 2];
            float y2 = boxes_data[i * 4 + 3];

            float sx = img_w / 640.0f;
            float sy = img_h / 480.0f;

            cv::Rect box(
                int(x1 * sx),
                int(y1 * sy),
                int((x2 - x1) * sx),
                int((y2 - y1) * sy)
            );

            box &= cv::Rect(0, 0, img_w, img_h);

            if (box.area() <= 0) continue;

            bboxes.push_back(box);
            confidences.push_back(conf);

            std::vector<cv::Point2f> lms(5);
            for (int j = 0; j < 5; j++) {
                float lx = lm_data[i * 10 + j * 2 + 0];
                float ly = lm_data[i * 10 + j * 2 + 1];
                lms[j] = cv::Point2f(lx * sx, ly * sy);
            }
            all_landmarks_temp.push_back(lms);
        }

        std::vector<int> idx;
        cv::dnn::NMSBoxes(bboxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, idx);

        std::vector<FaceData> faces;
        for (int i : idx) {
            FaceData f;
            f.bounding_box = bboxes[i];
            f.confidence = confidences[i];
            for (auto& p : all_landmarks_temp[i])
                f.landmarks.push_back(cv::Point2d(p));
            faces.push_back(f);
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


int main()
{
    int W = 640, H = 480;

    HeadPoseEstimator estimator("scrfd_2.5g_bnkps.onnx", W, H);

    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, W);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, H);

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

        estimator.process_frame(frame);

        cv::imshow("HeadPose", frame);
        if (cv::waitKey(1) == 27)
            break;
    }
}
