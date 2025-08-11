#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: cannot open camera\n";
        return -1;
    }

    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load("C:/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml")) {
        std::cerr << "Error: cascade\n";
        return -1;
    }

    cv::Mat frame, gray; //stores grayscale copy
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY); //convert grayscale
        cv::equalizeHist(gray, gray); //contrast

        std::vector<cv::Rect> faces;//detect face
        face_cascade.detectMultiScale(gray, faces, 1.1, 4, 0, cv::Size(30, 30));


        for (auto& face : faces)// draw box
        {
            cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("Face Detection", frame);

        if (cv::waitKey(10) == 27) break; //  to quit

    }
    return 0;
}