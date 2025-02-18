#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    Mat img = imread("C:/Users/AMD-PC/Desktop/Test.jpg");
    if (img.empty()) {
        cerr << "Ошибка: не удалось загрузить изображение!" << endl;
        return -1;
    }

    int width = img.cols;
    int height = img.rows;

    Rect roi1(0, 0, width / 2, height / 2);
    Rect roi2(width / 2, 0, width / 2, height / 2);
    Rect roi3(0, height / 2, width / 2, height / 2);
    Rect roi4(width / 2, height / 2, width / 2, height / 2);

    Mat part1 = img(roi1).clone();
    Mat part2, part3, part4;

    bitwise_not(img(roi2), part2);
    cvtColor(img(roi3), part3, COLOR_BGR2GRAY);
    cvtColor(part3, part3, COLOR_GRAY2BGR);
    part4 = Mat(img(roi4).size(), CV_8UC3, Scalar(255, 0, 0));

    Mat result = img.clone();
    part1.copyTo(result(roi1));
    part2.copyTo(result(roi2));
    part3.copyTo(result(roi3));
    part4.copyTo(result(roi4));

    imshow("Оригинал", img);
    imshow("Обработанное", result);
    waitKey(0);

    return 0;
}
