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
            letterBoxInfo lb = BackEndServiceHelper::letterbox(frame, img640, 640, 640);

            BackEndServiceHelper::fill_nchw_rgb(img640, input_buffer.data(), 640, 640);

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

                /*
                if (!faces640.empty()) {
                    // Pick the face we want to track (the one closest to center)
                    // Note: find_closest_face needs to be able to work with 640x640 input here
                    FaceData& target640 = faces640[0];

                    // --- START PFLD INSERTION ---

                    // 2. Create a square ROI from the face box (still in 640x640 space)
                    cv::Rect pfld_roi = get_square_box(target640.bounding_box, 640, 640);

                    // 3. Extract the face from the letterboxed image (zero-copy header)
                    cv::Mat face_crop = img640(pfld_roi);

                    // 4. Run your PFLD inference (outputs 98 points in 0.0 to 1.0 range)
                    std::vector<cv::Point2d> pfld_pts_raw = pfld_solver.detect(face_crop);

                    // 5. Replace the 5 SCRFD landmarks with the 98 PFLD landmarks
                    target640.landmarks.clear();
                    for (const auto& p : pfld_pts_raw) {
                        // Map from 0.0-1.0 local crop space -> 0-640 letterbox space
                        double lx = (p.x * pfld_roi.width) + pfld_roi.x;
                        double ly = (p.y * pfld_roi.height) + pfld_roi.y;
                        target640.landmarks.emplace_back(lx, ly);
                    }

                    // --- END PFLD INSERTION ---
                }
                */

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