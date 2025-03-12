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

    Mat imgGray;
    cvtColor(img, imgGray, COLOR_BGR2GRAY);

    Mat imgBlurred;
    GaussianBlur(imgGray, imgBlurred, Size(5, 5), 0);

    imshow("Original", img);
    imshow("Gray", imgGray);
    imshow("Gaussian Blur", imgBlurred);
    imwrite("Gray.jpg", imgGray);
    imwrite("Blur.jpg", imgBlurred);

    waitKey(0);
    return 0;
}
