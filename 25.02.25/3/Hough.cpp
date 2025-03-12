#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    Mat img = imread("Test.jpg");
    if (img.empty()) {
        cerr << "Ошибка: не удалось загрузить изображение!" << endl;
        return -1;
    }

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    Mat gray3;
    cvtColor(gray, gray3, COLOR_GRAY2BGR);

    GaussianBlur(gray, gray, Size(5, 5), 2);

    Mat edges;
    Canny(gray, edges, 50, 150);

    vector<Vec2f> lines;
    HoughLines(edges, lines, 1, CV_PI / 180, 160);

    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0], theta = lines[i][1];
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
        Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));
        line(gray3, pt1, pt2, Scalar(0, 0, 255), 2);
    }

    vector<Vec3f> circles;
    HoughCircles(gray, circles, HOUGH_GRADIENT, 1, gray.rows / 4, 100, 40, 30, 100);

    for (size_t i = 0; i < circles.size(); i++) {
        Vec3i c = circles[i];
        Point center(c[0], c[1]);
        int radius = c[2];
        circle(gray3, center, radius, Scalar(255, 0, 0), 2);  
        circle(gray3, center, 3, Scalar(0, 255, 0), -1);  
    }
    imshow("Isxod", img);
    imshow("Опр", gray3);
    imwrite("Круг и линии.jpg", gray3);
    waitKey(0);

    return 0;
}
