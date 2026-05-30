#include "HeadPoseSolver.h"
#include "BackendServiceHelper.h"
#include "BackendHelperModels.h"
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

int main()
{
    //obsolete need to remove
    static FacePoseController controller;

    // SCRFD model config
    std::int64_t batch_scfrd = 1;
    std::int64_t numchannels_scfrd = 3;
    std::int64_t width_scfrd = 640;
    std::int64_t height_scfrd = 640;
    std::vector<int64_t> input_shape_scfrd = { batch_scfrd, numchannels_scfrd, height_scfrd, width_scfrd };
    size_t input_tensor_size_scrfd = numchannels_scfrd * height_scfrd * width_scfrd;

    // 1k3d68 model config
    std::int64_t batch_1k3d68 = 1;
    std::int64_t numchannels_1k3d68 = 3;
    std::int64_t width_1k3d68 = 192;
    std::int64_t height_1k3d68 = 192;
    std::vector<int64_t> input_shape_1k3d68 = { batch_1k3d68, numchannels_1k3d68, height_1k3d68, width_1k3d68 };
    size_t input_tensor_size_1k3d68 = numchannels_1k3d68 * height_1k3d68 * width_1k3d68;

    // ORT setup
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "scrfd & 1k3d68");

    Ort::SessionOptions session_options_scrfd;
    Ort::SessionOptions session_options_1k3d68;
    // By not setting this, ORT automatically manages cores and affinities
    session_options_scrfd.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options_1k3d68.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session_scrfd(env, L"det_10g.onnx", session_options_scrfd);
    Ort::Session session_1k3d68(env, L"1k3d68.onnx", session_options_1k3d68);

    std::vector<float> input_buffer_scfrd(input_tensor_size_scrfd);
    std::vector<float> input_buffer_1k3d68(input_tensor_size_1k3d68);

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Input/output names
    Ort::AllocatorWithDefaultOptions allocator;

    auto input_name_scfrd = session_scrfd.GetInputNameAllocated(0, allocator);
    std::vector<const char*> input_names_scrfd = { input_name_scfrd.get() };

    auto input_name_1k3d68 = session_1k3d68.GetInputNameAllocated(0, allocator);
    std::vector<const char*> input_names_1k3d68 = { input_name_1k3d68.get() };

    std::vector<const char*> output_names_scfrd;
    std::vector<Ort::AllocatedStringPtr> output_name_allocs_scfrd;
    for (size_t i = 0; i < session_scrfd.GetOutputCount(); ++i) {
        output_name_allocs_scfrd.push_back(session_scrfd.GetOutputNameAllocated(i, allocator));
        output_names_scfrd.push_back(output_name_allocs_scfrd.back().get());
    }

    std::vector<const char*> output_names_1k3d68;
    std::vector<Ort::AllocatedStringPtr> output_name_allocs_1k3d68;
    for (size_t i = 0; i < session_1k3d68.GetOutputCount(); i++)
    {
        output_name_allocs_1k3d68.push_back(session_1k3d68.GetOutputNameAllocated(i, allocator));
        output_names_1k3d68.push_back(output_name_allocs_1k3d68.back().get());
    }

    HeadPoseSolver solver(640, 480);

    // OpenCV camera
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
        letterBoxInfo lb = BackEndServiceHelper::letterbox(frame, img640, 640, 640);
        BackEndServiceHelper::fill_nchw_rgb_float(img640, input_buffer_scfrd.data(), 640, 640);

        Ort::Value input_tensor_scfrd = Ort::Value::CreateTensor<float>(
            memory_info,
            input_buffer_scfrd.data(),
            input_buffer_scfrd.size(),
            input_shape_scfrd.data(),
            input_shape_scfrd.size()
        );

        try {
            auto output_scfrd = session_scrfd.Run(
                Ort::RunOptions{ nullptr },
                input_names_scrfd.data(),
                &input_tensor_scfrd,
                1,
                output_names_scfrd.data(),
                output_names_scfrd.size()
            );

            auto faces640 = solver.parse_scrfd_ort(output_scfrd, 640, 640);
            auto faces = BackEndServiceHelper::unletterbox_faces(faces640, lb, frame.cols, frame.rows);

            if (!faces.empty()) {
                FaceData target = solver.find_closest_face(faces);

                // Draw landmarks
                for (int i = 0; i < (int)target.landmarks.size(); ++i) {
                    cv::circle(frame, target.landmarks[i], 2, cv::Scalar(0, 255, 0), -1);
                    cv::putText(frame, std::to_string(i), target.landmarks[i],
                        cv::FONT_HERSHEY_PLAIN, 0.6, cv::Scalar(255, 255, 0), 1);
                }

                cv::Mat crop192;
                cv::Mat M = BackEndServiceHelper::cropFaceFor1k3d68(frame, target, crop192, input_buffer_1k3d68.data());

                Ort::Value input_tensor_1k3d68 = Ort::Value::CreateTensor<float>(
                    memory_info,
                    input_buffer_1k3d68.data(),
                    input_buffer_1k3d68.size(),
                    input_shape_1k3d68.data(),
                    input_shape_1k3d68.size()
                );

                auto output_1k3d68 = session_1k3d68.Run(
                    Ort::RunOptions{ nullptr },
                    input_names_1k3d68.data(),
                    &input_tensor_1k3d68,
                    1,
                    output_names_1k3d68.data(),
                    output_names_1k3d68.size()
                );

                auto landmarks68 = solver.parse_1k3d68_ort(output_1k3d68, M);
                solver.solveAffine(landmarks68);

                cv::rectangle(frame, target.bounding_box, cv::Scalar(0, 255, 0), 2);
                char text[128];
                std::snprintf(text, sizeof(text), "Pitch: %5.1f  Yaw: %5.1f  Roll: %5.1f",
                solver.get_pitch(), solver.get_yaw(), solver.get_roll());
                cv::putText(frame, text, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
            }
        }
        catch (const Ort::Exception& e) {
            spdlog::error("ORT ERROR: {}", e.what());
            return -1;
        }

        cv::imshow("Camera", frame);
        if (cv::waitKey(1) == 27)
            break;
    }

    return 0;
}