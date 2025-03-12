#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    Mat img = imread("Test.jpg");
    if (img.empty()) {
        cout << "Ошибка загрузки изображения!" << endl;
        return -1;
    }

    Mat imgHSV, imgLab, imgYUV, imgXYZ, imgGray;

    cvtColor(img, imgHSV, COLOR_BGR2HSV);
    cvtColor(img, imgLab, COLOR_BGR2Lab);
    cvtColor(img, imgYUV, COLOR_BGR2YUV);
    cvtColor(img, imgXYZ, COLOR_BGR2XYZ);
    cvtColor(img, imgGray, COLOR_BGR2GRAY);

    imshow("Original", img);
    imshow("HSV", imgHSV);
    imshow("Lab", imgLab);
    imshow("YUV", imgYUV);
    imshow("XYZ", imgXYZ);
    imshow("Gray", imgGray);
    imwrite("HSV.jpg", imgHSV);
    imwrite("Lab.jpg", imgLab);
    imwrite("YUV.jpg", imgYUV);
    imwrite("XYZ.jpg", imgXYZ);
    imwrite("Gray.jpg", imgGray);
    waitKey(0);
    return 0;
}
