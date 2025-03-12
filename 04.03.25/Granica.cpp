#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

string getShapeName(const vector<Point>& contour) {
    vector<Point> approx;
    approxPolyDP(contour, approx, 0.02 * arcLength(contour, true), true);
    int vertices = (int)approx.size();

    if (vertices == 3) return "Triangle";
    else if (vertices == 4) {
        Rect rect = boundingRect(approx);
        double aspectRatio = (double)rect.width / rect.height;
        return (aspectRatio > 0.9 && aspectRatio < 1.1) ? "Square" : "Rectangle";
    }
    else {
       double perimeter = arcLength(contour, true);
       double area = contourArea(contour);

       double circularity = (4 * CV_PI * area) / (perimeter * perimeter);

       if (circularity > 0.8)
        return "Circle";
    }
    return "Unknown";
}

int main() {
    string imagePath = "C:/Users/AMD-PC/Desktop/img.png";
    Mat img = imread(imagePath);
    copyMakeBorder(img, img, 10, 10, 10, 10, BORDER_CONSTANT, Scalar(0, 0, 0));
    if (img.empty()) {
        cerr << "Îøèáêà: " << imagePath << endl;
        return -1;
    }

    Mat gray, blurred, edges;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    medianBlur(gray, blurred, 5);
    Canny(blurred, edges, 50, 150);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat output = Mat::zeros(img.size(), CV_8UC3);

    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (area < 150) continue;

        string shapeName = getShapeName(contour);

        Moments M = moments(contour);
        if (M.m00 != 0) {
            int cx = int(M.m10 / M.m00);
            int cy = int(M.m01 / M.m00);

            int fontFace = FONT_HERSHEY_SIMPLEX;
            double fontScale = 0.7;
            int thickness = 2;
            int baseline = 0;
            Size textSize = getTextSize(shapeName, fontFace, fontScale, thickness, &baseline);
            Point textOrg(cx - textSize.width / 2, cy + textSize.height / 2);

            drawContours(output, vector<vector<Point>>{contour}, -1, Scalar(255, 255, 255), 2);
            putText(output, shapeName, textOrg, fontFace, fontScale, Scalar(255, 255, 255), thickness);
        }
    }
    Rect roi(10, 10, img.cols - 20, img.rows - 20);
    img = img(roi);
    imshow("Contours", output);
    imwrite("Result.jpg", output);
    waitKey(0);
    return 0;
}
