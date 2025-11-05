#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <vector>
#include <future>
#include <tuple>
#include <iostream>
#include <numeric>

struct FaceData {
    cv::Rect2d bounding_box;
    std::vector<cv::Point2d> landmarks;
    float confidence;
};

struct PoseData {
    std::vector<cv::Point2d> image_points;
    cv::Mat rotation_vector;
    cv::Mat translation_vector;
};

class HeadPoseEstimator {
public:
    HeadPoseEstimator(const std::string& scrfd_model_path, int width, int height)
        : frame_width(width), frame_height(height) 
    {
        face_detector = cv::dnn::readNetFromONNX(scrfd_model_path);
        face_detector.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        face_detector.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

        initLogger();
        initCameraMatrix();
        dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);
    }

    std::vector<cv::Mat> get_cpu_output_from_gpu(const std::vector<cv::Mat>& gpu_output_blobs) {
        std::vector<cv::Mat> cpu_output_blobs;
        for (const cv::Mat& gpu_wrapper : gpu_output_blobs) {
            
            // 1. Download data: GPU -> CPU
            cv::Mat raw_cpu_data;
            gpu_wrapper.copyTo(raw_cpu_data);

            if (raw_cpu_data.total() > 0) {
                
                //Flatten to a single row (1 channel, 1 row) This correctly sets the cv::Mat header required for safe .data pointer access.
                cv::Mat flat_cpu_blob = raw_cpu_data.reshape(1, 1); 
                cpu_output_blobs.push_back(flat_cpu_blob);
            } else {
                cpu_output_blobs.push_back(cv::Mat());
            }
        }
        return cpu_output_blobs;
    }

    void process_frame(cv::Mat& frame) {
        cv::cuda::GpuMat gpu_frame;
        gpu_frame.upload(frame);
        cv::cuda::GpuMat blob_gpu;
        cv::dnn::blobFromImage(gpu_frame, blob_gpu, 1.0, cv::Size(640, 480), cv::Scalar(104, 117, 123), false, false);

        face_detector.setInput(blob_gpu);
        std::vector<cv::Mat> output_blobs; // These Mats wrap GPU memory
        face_detector.forward(output_blobs, face_detector.getUnconnectedOutLayersNames());

        std::vector<cv::Mat> cpu_output_blobs = get_cpu_output_from_gpu(output_blobs);
        
        std::vector<FaceData> faces = parse_scrfd_output(cpu_output_blobs, frame.cols, frame.rows);

        if (!faces.empty()) {
            FaceData main_face = find_closest_face(faces);
            std::vector<cv::Point2d> image_points = main_face.landmarks;

            cv::Mat rvec, tvec;
            cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rvec, tvec);

            cv::rectangle(frame, main_face.bounding_box, cv::Scalar(0, 255, 0), 2);
            
            // Draw the head pose axes
            draw_pose_axis(frame, rvec, tvec);
        }
    }

    cv::Vec3d getYawPitchRoll(const cv::Mat& rvec) {
        cv::Mat rotation_matrix;
        cv::Rodrigues(rvec, rotation_matrix);

        double sy = sqrt(rotation_matrix.at<double>(0, 0) * rotation_matrix.at<double>(0, 0) +
            rotation_matrix.at<double>(1, 0) * rotation_matrix.at<double>(1, 0));
        bool singular = sy < 1e-6;

        double x, y, z;
        if (!singular) {
            x = atan2(rotation_matrix.at<double>(2, 1), rotation_matrix.at<double>(2, 2));
            y = atan2(-rotation_matrix.at<double>(2, 0), sy);
            z = atan2(rotation_matrix.at<double>(1, 0), rotation_matrix.at<double>(0, 0));
        }
        else {
            x = atan2(-rotation_matrix.at<double>(1, 2), rotation_matrix.at<double>(1, 1));
            y = atan2(-rotation_matrix.at<double>(2, 0), sy);
            z = 0;
        }

        return cv::Vec3d(x, y, z);
    }

    std::vector<FaceData> parse_scrfd_output(const std::vector<cv::Mat>& output_blobs, int img_w, int img_h) {
        const float CONFIDENCE_THRESHOLD = 0.5f;
        const float NMS_THRESHOLD = 0.45f;
        
        if (output_blobs.size() < 3 || output_blobs[0].empty() || output_blobs[1].empty() || output_blobs[2].empty()) {
            return {};
        }

        const cv::Mat& boxes_blob = output_blobs[0];
        const cv::Mat& scores_blob = output_blobs[1];
        const cv::Mat& landmarks_blob = output_blobs[2];
        std::vector<cv::Rect> bboxes;
        std::vector<float> confidences;
        std::vector<std::vector<cv::Point2f>> all_landmarks_temp;
        
        int num_predictions = scores_blob.cols / 2; 

        const float* scores_data = (const float*)scores_blob.data;
        const float* boxes_data = (const float*)boxes_blob.data;
        const float* landmarks_data = (const float*)landmarks_blob.data;

        for (int i = 0; i < num_predictions; ++i) {
            float face_score = scores_data[i * 2 + 1];

            if (face_score > CONFIDENCE_THRESHOLD) {
                float x1 = boxes_data[i * 4 + 0];
                float y1 = boxes_data[i * 4 + 1];
                float x2 = boxes_data[i * 4 + 2];
                float y2 = boxes_data[i * 4 + 3];
                float scale_x = (float)img_w / 640.0f;
                float scale_y = (float)img_h / 480.0f;
                int x = static_cast<int>(x1 * scale_x);
                int y = static_cast<int>(y1 * scale_y);
                int width = static_cast<int>((x2 - x1) * scale_x);
                int height = static_cast<int>((y2 - y1) * scale_y);
                cv::Rect box(x, y, width, height);
                box &= cv::Rect(0, 0, img_w, img_h);

                if (box.area() > 0) {
                    bboxes.push_back(box);
                    confidences.push_back(face_score);
                    std::vector<cv::Point2f> lms(5);
                    for (int j = 0; j < 5; ++j) {
                        float lx = landmarks_data[i * 10 + j * 2 + 0];
                        float ly = landmarks_data[i * 10 + j * 2 + 1];
                        lms[j] = cv::Point2f(lx * scale_x, ly * scale_y);
                    }
                    all_landmarks_temp.push_back(lms);
                }
            }
        }
        std::vector<int> indices;
        cv::dnn::NMSBoxes(bboxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);
        std::vector<FaceData> faces_out;
        for (int i : indices) {
            FaceData face;
            face.confidence = confidences[i];
            face.bounding_box = cv::Rect2d(bboxes[i]);
            for (const auto& p_f : all_landmarks_temp[i]) {
                face.landmarks.push_back(cv::Point2d(p_f.x, p_f.y));
            }
            faces_out.push_back(face);
        }
        return faces_out;
    }

    FaceData find_closest_face(const std::vector<FaceData>& faces) {
        if (faces.empty()) {
            return {}; 
        }

        cv::Point2d image_center(frame_width / 2.0, frame_height / 2.0);
        double min_distance_sq = std::numeric_limits<double>::max();
        FaceData best_face;

        for (const auto& face : faces) {
            cv::Point2d face_center(
                face.bounding_box.x + face.bounding_box.width / 2.0,
                face.bounding_box.y + face.bounding_box.height / 2.0
            );

            double dx = face_center.x - image_center.x;
            double dy = face_center.y - image_center.y;
            double distance_sq = dx * dx + dy * dy;

            if (distance_sq < min_distance_sq) {
                min_distance_sq = distance_sq;
                best_face = face;
            }
        }
        return best_face;
    }

private:
    cv::dnn::Net face_detector;
    int frame_width, frame_height;
    cv::Mat camera_matrix, dist_coeffs;

    std::vector<cv::Point3d> model_points{
        {0.0, 0.0, 0.0},{225.0, 170.0, -135.0}, {-225.0, 170.0, -135.0},{150.0, -150.0, -125.0},{-150.0, -150.0, -125.0}
    };

    void initCameraMatrix() {
        double focal_length = frame_width * 1.2;
        cv::Point2d center(frame_width / 2.0, frame_height / 2.0);
        camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, center.x,
            0, focal_length, center.y,
            0, 0, 1);
        spdlog::info("Camera matrix initialized.");
    }

    void initLogger() {
        try {
            auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("headpose.log", true);
            std::vector<spdlog::sink_ptr> sinks{ console_sink, file_sink };
            auto logger = std::make_shared<spdlog::logger>("headpose_logger", sinks.begin(), sinks.end());
            spdlog::set_default_logger(logger);
            spdlog::set_level(spdlog::level::info);
            spdlog::info("Logger started.");
        }
        catch (const spdlog::spdlog_ex& ex) {
            std::cout << "Log failed to start: " << ex.what() << std::endl;
        }
    }

    void draw_pose_axis(cv::Mat& frame, const cv::Mat& rvec, const cv::Mat& tvec) {
        std::vector<cv::Point3d> axis_points = {
            {0, 0, 0},      // Origin (Nose Tip)
            {100, 0, 0},    // X-axis (Red - Yaw/Turn Right)
            {0, 100, 0},    // Y-axis (Green - Pitch/Look Up)
            {0, 0, 100}     // Z-axis (Blue - Roll/Head Tilt Forward)
        };

        std::vector<cv::Point2d> image_points;
        cv::projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs, image_points);

        if (image_points.size() == 4) {
            cv::Point p0 = image_points[0]; // Origin (Nose Tip)
            cv::line(frame, p0, image_points[1], cv::Scalar(0, 0, 255), 3); // X-axis (Red)
            cv::line(frame, p0, image_points[2], cv::Scalar(0, 255, 0), 3); // Y-axis (Green)
            cv::line(frame, p0, image_points[3], cv::Scalar(255, 0, 0), 3); // Z-axis (Blue)
        }
    }
};

int main() {
    int frame_width = 640, frame_height = 480;
    HeadPoseEstimator estimator("scrfd_2.5g_bnkps.onnx", frame_width, frame_height); 

    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, frame_width); 
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, frame_height);

    if (!cap.isOpened()) {
        spdlog::error("Cannot open camera.");
        return -1;
    }
    spdlog::info("Camera opened. Actual frame size: {}x{}", cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        estimator.process_frame(frame);

        cv::imshow("Head Prediction", frame);
        if (cv::waitKey(1) == 27) break;
    }

    return 0;
}