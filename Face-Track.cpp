#include "HeadPoseSolver.h"
#include "BackendServiceHelper.h"
#include "BackendHelperModels.h"
#include "Parser.h"
#include "ONNXModel.h"
#include "IFrameSource.h"
#include "FrameSourceFactory.h"

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <spdlog/spdlog.h>

#include <vector>
#include <cstdint>
#include <cstdio>
#include <string>



#define TICK(name)\
    auto t_##name = std::chrono::high_resolution_clock::now();

#define TOCK(name) \
    spdlog::info("{}: {} ms", #name, \
    std::chrono::duration_cast<std::chrono::milliseconds>( \
        std::chrono::high_resolution_clock::now() - t_##name).count());

int main()
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "scrfd & 1k3d68");

    ONNXModel scrfd(env, L"det_10g.onnx", { 1, 3, 640, 640 });
    ONNXModel landmark(env, L"1k3d68.onnx", { 1, 3, 192, 192 });

    HeadPoseSolver solver(640, 480);

    std::unique_ptr<IFrameSource> source = makeFrameSource(SourceType::Webcam, "");
    cv::Mat frame;
        
    while (source -> read(frame))
    {

        try {
            cv::Mat img640;
            letterBoxInfo lb = LetterBoxGeometry::letterbox(frame, img640, 640, 640);

            ModelInput::pack_nchw_rgb(img640, scrfd.input_data(), 0.0f, 1.0f / 255.0f);

            auto out_scrfd = scrfd.run();
            auto faces640 = Parser::parse_scrfd_ort(out_scrfd, 640, 640);
            auto faces = LetterBoxGeometry::unletterbox_faces(faces640, lb, frame.cols, frame.rows);

            if (!faces.empty()) {
                FaceData target = solver.find_closest_face(faces);

                // Draw landmarks
                for (int i = 0; i < (int)target.landmarks.size(); ++i) {
                    cv::circle(frame, target.landmarks[i], 2, cv::Scalar(0, 255, 0), -1);
                    cv::putText(frame, std::to_string(i), target.landmarks[i],
                        cv::FONT_HERSHEY_PLAIN, 0.6, cv::Scalar(255, 255, 0), 1);
                }

                cv::Mat crop192;
                cv::Mat M = ModelInput::cropFaceFor1k3d68(frame, target, crop192, landmark.input_data());
                
                auto out_landmark = landmark.run();
                auto landmarks68 = Parser::parse_1k3d68_ort(out_landmark, M);

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