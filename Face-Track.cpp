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
        static FacePoseController controller;

        // Ort scrfd model config 
        std::int64_t batch_scfrd = 1;
        std::int64_t numchannels_scfrd = 3;
        std::int64_t width_scfrd = 640;
        std::int64_t height_scfrd = 640;
        std::vector <int64_t> input_shape_scfrd = { batch_scfrd, numchannels_scfrd, height_scfrd, width_scfrd };
        size_t input_tensor_size_scrfd = numchannels_scfrd * height_scfrd * width_scfrd;

        std::int64_t batch_pfld = 1;
        std::int64_t numchannels_pfld = 3;
        std::int64_t width_pfld = 128;
        std::int64_t height_pfld = 128;
        std::vector <int64_t> input_shape_pfld = { batch_pfld, numchannels_pfld, height_pfld, width_pfld };
        size_t input_tesor_size_pfld = numchannels_pfld * height_pfld * width_pfld;

        // Ort setup
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "scrfd");

        Ort::SessionOptions session_options_scrfd;
        //by not setting this option ort autmaticly manages cores and affinitisez them
        //session_options.SetIntraOpNumThreads(1);
        session_options_scrfd.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        Ort::SessionOptions session_options_pfld;
        session_options_pfld.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        Ort::Session session_scrfd(env, L"scrfd_model.onnx", session_options_scrfd);
        Ort::Session session_pfld(env, L"landmark_detection_model.onnx", session_options_pfld);


        std::vector <float> input_buffer_scfrd(input_tensor_size_scrfd);
        std::vector <uint8_t> input_buffer_pfld(input_tesor_size_pfld);

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);//check later if OrtMemType... will need to be cuda specific

        // input/output names
        Ort::AllocatorWithDefaultOptions allocator;

        auto input_name_scfrd = session_scrfd.GetInputNameAllocated(0, allocator);
        std::vector<const char*> input_names_scrfd = { input_name_scfrd.get() };

        auto input_name_pfld = session_pfld.GetInputNameAllocated(0, allocator);
        std::vector<const char*> input_names_pfld = { input_name_pfld.get()};


        auto input_name_scfrd_alloc = session_scrfd.GetInputNameAllocated(0, allocator);
        std::vector<const char*> output_names_scfrd;
        std::vector<Ort::AllocatedStringPtr> output_name_allocs_scfrd;

        for (size_t i = 0; i < session_scrfd.GetOutputCount(); ++i) {
            output_name_allocs_scfrd.push_back(session_scrfd.GetOutputNameAllocated(i, allocator));
            output_names_scfrd.push_back(output_name_allocs_scfrd.back().get());
        }

        auto input_name_pfld_alloc = session_pfld.GetInputNameAllocated(0, allocator);
        std::vector<const char*> output_names_pfld;
        std::vector<Ort::AllocatedStringPtr> output_name_allocs_pfld;

        for (size_t i = 0; i < session_pfld.GetOutputCount(); ++i) {
            output_name_allocs_pfld.push_back(session_pfld.GetOutputNameAllocated(i, allocator));
            output_names_pfld.push_back(output_name_allocs_pfld.back().get());
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

                
                if (!faces640.empty()) {
                    FaceData& target640 = faces640[0];

                    cv::Rect roi_rect = BackEndServiceHelper::get_pfld_roi(target640.bounding_box, 640);
                    cv::Mat face_roi_view = img640(roi_rect);

                    // Resize that specific ROI to PFLD's expected 112x112
                    cv::Mat pfld_input;  
                    cv::resize(face_roi_view, pfld_input, cv::Size(128, 128));

                    BackEndServiceHelper::fill_nchw_rgb_uint8_t(pfld_input, input_buffer_pfld.data(), 128, 128);

                    Ort::Value input_tensor_pfld = Ort::Value::CreateTensor<uint8_t>(
                        memory_info,
                        input_buffer_pfld.data(),
                        input_buffer_pfld.size(),
                        input_shape_pfld.data(),
                        input_shape_pfld.size()
                    );


                    // Run PFLD
                    try {
                        auto output_pfld = session_pfld.Run(
                            Ort::RunOptions{ nullptr },
                            input_names_pfld.data(),
                            &input_tensor_pfld,
                            1,
                            output_names_pfld.data(),
                            1);

                        Ort::Value& landmarks_tensor = output_pfld.at(0);
                        float* raw_values = landmarks_tensor.GetTensorMutableData<float>();
                        std::vector<cv::Point2d> local_landmarks = BackEndServiceHelper::point2d_converstion(raw_values);

                        target640.landmarks.clear();

                        for (int i = 0; i < 98; ++i) {
                            float x_norm = raw_values[i * 2];
                            float y_norm = raw_values[i * 2 + 1];
                            double global_x = (x_norm * roi_rect.width) + roi_rect.x;
                            double global_y = (y_norm * roi_rect.height) + roi_rect.y;

                            target640.landmarks.emplace_back(global_x, global_y);
                        }
                    }
                    catch (const Ort::Exception& e) {// logger yap
                        std::cerr << "PFLD Inference failed: " << e.what() << std::endl;
                    }
                }
                

                auto faces = BackEndServiceHelper::unletterbox_faces(faces640, lb, frame.cols, frame.rows);


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