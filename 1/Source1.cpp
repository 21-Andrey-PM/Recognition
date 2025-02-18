#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    string imagePath = "C:/Users/AMD-PC/Desktop/Test.jpg";

    Mat image = imread(imagePath);

    if (image.empty()) {
        std::cerr << "Ошибка: не удалось загрузить изображение!" << std::endl;
        return -1;
    }

   line(image, Point(50, 50), Point(200, 50), Scalar(0, 255, 0), 2);

    rectangle(image, Point(50, 80), Point(200, 150), Scalar(255, 0, 0), 2);

    circle(image, Point(125, 250), 50, Scalar(0, 0, 255), 2);

    ellipse(image, Point(125, 350), Size(60, 30), 30, 0, 360, Scalar(255, 255, 0), 2);

    vector<Point> points = { Point(50, 400), Point(100, 450), Point(150, 400), Point(100, 350) };
    polylines(image, points, true, Scalar(255, 0, 255), 2);

    putText(image, "Solo leveling", Point(50, 480), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 2);

    imshow("Edited Image", image);
    waitKey(0);

    return 0;
}
