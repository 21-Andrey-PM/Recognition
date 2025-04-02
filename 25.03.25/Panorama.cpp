#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main() {
    Mat img1 = imread("img1.png");
    Mat img2 = imread("img2.png");

    if (img1.empty() || img2.empty()) {
        cout << "Ошибка" << endl;
        return -1;
    }

    Mat gray1, gray2;
    cvtColor(img1, gray1, COLOR_BGR2GRAY);
    cvtColor(img2, gray2, COLOR_BGR2GRAY);

    Ptr<SIFT> detector = SIFT::create();

    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute(gray1, noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(gray2, noArray(), keypoints2, descriptors2);

    BFMatcher matcher(NORM_L2);
    vector<vector<DMatch>> knn_matches;
    matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);

    vector<DMatch> good_matches;
    for (const auto& match : knn_matches) {
        if (match[0].distance < 0.75f * match[1].distance) {
            good_matches.push_back(match[0]);
        }
    }

    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches);
    imshow("Good Matches", img_matches);
    waitKey(0);

    if (good_matches.size() < 4) {
        cout << "Недостаточно совпадений" << endl;
        return -1;
    }

    vector<Point2f> points1, points2;
    for (const auto& match : good_matches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }

    Mat H = findHomography(points2, points1, RANSAC);

    Mat result;
    warpPerspective(img2, result, H, Size(img1.cols * 2, img1.rows));

    img1.copyTo(result(Rect(0, 0, img1.cols, img1.rows)));

    Mat gray;
    cvtColor(result, gray, COLOR_BGR2GRAY);
    threshold(gray, gray, 1, 255, THRESH_BINARY);
    Rect roi = boundingRect(gray);
    Mat cropped_result = result(roi);

    imshow("Cropped Panorama", cropped_result);
    waitKey(0);

    imwrite("panorama_result.jpg", cropped_result);

    return 0;
}