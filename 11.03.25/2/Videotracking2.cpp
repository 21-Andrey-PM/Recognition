#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define COLOR_TRIANGLE Scalar(128, 0, 128)    
#define COLOR_RECTANGLE Scalar(255, 69, 0)    
#define COLOR_SQUARE Scalar(0, 128, 128)      
#define COLOR_CIRCLE Scalar(255, 215, 0)      
#define COLOR_UNKNOWN Scalar(169, 169, 169)   

string detectShape(const vector<Point>& contour) {
    double peri = arcLength(contour, true);
    vector<Point> approx;
    approxPolyDP(contour, approx, 0.027 * peri, true);

    int vertices = approx.size();

    if (vertices == 3) return "Triangle";
    if (vertices == 4) {
        Rect rect = boundingRect(approx);
        double aspectRatio = (double)rect.width / rect.height;
        if (aspectRatio > 0.9 && aspectRatio < 1.1) {
            return "Square";
        }
        else {
            return "Rectangle";
        }
    }

    double area = contourArea(contour);
    double circularity = (4 * CV_PI * area) / (peri * peri);

    Rect boundingBox = boundingRect(contour);
    double aspectRatio = (double)boundingBox.width / boundingBox.height;

    if (vertices > 6 && circularity > 0.8 && aspectRatio > 0.9 && aspectRatio < 1.1)
        return "Circle";

    return "Unknown";
}

int main() {
    VideoCapture cap("video.mp4");
    if (!cap.isOpened()) {
        cerr << "Error!" << endl;
        return -1;
    }

    int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(CAP_PROP_FPS);

    VideoWriter videoWriter("output.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(frame_width, frame_height));
    if (!videoWriter.isOpened()) {
        cerr << "Error" << endl;
        return -1;
    }

    while (true) {
        Mat frame, gray, blurred, thresh;
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, blurred, Size(5, 5), 0);
        threshold(blurred, thresh, 225, 255, THRESH_BINARY_INV);

        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(thresh, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        for (const auto& contour : contours) {
            double area = contourArea(contour);
            if (area < 5 || area > 0.5 * frame.rows * frame.cols) continue;

            string shape = detectShape(contour);
            Scalar color;
            if (shape == "Triangle") color = COLOR_TRIANGLE;
            else if (shape == "Rectangle") color = COLOR_RECTANGLE;
            else if (shape == "Square") color = COLOR_SQUARE;
            else if (shape == "Circle") color = COLOR_CIRCLE;
            else color = COLOR_UNKNOWN;

            drawContours(frame, vector<vector<Point>>{contour}, -1, color, 2);
            Moments m = moments(contour);
            if (m.m00 != 0) {
                int cx = (int)(m.m10 / m.m00);
                int cy = (int)(m.m01 / m.m00);
                putText(frame, shape, Point(cx, cy), FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
            }
        }

        videoWriter.write(frame);
        imshow("Shapes", frame);
        if (waitKey(30) == 27) break;
    }

    cap.release();
    videoWriter.release();
    destroyAllWindows();
    return 0;
}