#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    VideoCapture cap("C:/Users/AMD-PC/Desktop/video.mp4");
    if (!cap.isOpened()) {
        cout << "Ошибка: не удалось открыть видео!" << endl;
        return -1;
    }

    int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    VideoWriter outputVideo("utput_video.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'),
        cap.get(CAP_PROP_FPS), Size(frame_width, frame_height));

    Mat frame, gray, edges;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);
        Canny(gray, edges, 50, 150);

        findContours(edges, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
        for (size_t i = 0; i < contours.size(); i++) {
            vector<Point> approx;
            double peri = arcLength(contours[i], true);
            approxPolyDP(contours[i], approx, 0.02 * peri, true);

            if (approx.size() == 4 && isContourConvex(approx)) {
                Rect boundRect = boundingRect(approx);

                if (hierarchy[i][2] != -1) {
                    if (boundRect.area() > 1000) {
                        Mat mask = Mat::zeros(frame.size(), CV_8UC1);
                        fillConvexPoly(mask, approx, Scalar(255));
                        Scalar meanColor = mean(frame, mask);

                        if (meanColor[0] > 130 && meanColor[1] > 130 && meanColor[2] > 130) {
                            drawContours(frame, vector<vector<Point>>{approx}, -1, Scalar(0, 255, 0), 3);
                            Point center(boundRect.x + boundRect.width / 2, boundRect.y + boundRect.height / 2);
                            putText(frame, "Square", center, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
                        }
                    }
                }
            }
        }

        outputVideo.write(frame);
        imshow("Video", frame);
        if (waitKey(30) >= 0) break;
    }

    cap.release();
    outputVideo.release();
    destroyAllWindows();
    return 0;
}
