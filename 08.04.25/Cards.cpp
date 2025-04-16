#include <filesystem>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;
namespace fs = filesystem;

vector<DMatch> ratioTest(const vector<vector<DMatch>>& matches12, double ratio) {
    vector<DMatch> good_matches;
    for (const auto& match : matches12) {
        if (match.size() >= 2 && match[0].distance < ratio * match[1].distance) {
            good_matches.push_back(match[0]);
        }
    }
    return good_matches;
}

bool parseImage(const fs::path& img_path, Ptr<SIFT>& sift, vector<Mat>& images, vector<vector<KeyPoint>>& keypoints, vector<Mat>& descriptors, vector<string>& names) {
    Mat image = imread(img_path.string());
    if (image.empty()) {
        cout << "Ошибка парсинга: " << img_path << endl;
        return false;
    }

    vector<KeyPoint> keypoint;
    Mat desc;
    sift->detectAndCompute(image, noArray(), keypoint, desc);

    names.push_back(img_path.stem().string());
    images.push_back(image);
    keypoints.push_back(keypoint);
    descriptors.push_back(desc);

    cout << "Парсинг: " << img_path << endl;
    return true;
}

Mat matchImages(const Mat& target, const vector<Mat>& images, const vector<vector<KeyPoint>>& keypoints, const vector<Mat>& descriptors, const vector<string>& names, Ptr<SIFT>& sift) {
    Mat imgMatches = target.clone();
    BFMatcher matcher(NORM_L2);

    vector<KeyPoint> targetKeypoints;
    Mat targetDescriptors;
    sift->detectAndCompute(target, noArray(), targetKeypoints, targetDescriptors);

    for (size_t i = 0; i < names.size(); i++) {
        if (descriptors[i].empty() || targetDescriptors.empty())
            continue;

        vector<vector<DMatch>> matches;
        matcher.knnMatch(descriptors[i], targetDescriptors, matches, 2);

        auto good_matches = ratioTest(matches, 0.75);

        if (good_matches.size() < 4)
            continue;

        vector<Point2f> points_sample, points_target;
        for (int j = 0; j < good_matches.size(); j++) {
            points_sample.push_back(keypoints[i][good_matches[j].queryIdx].pt);
            points_target.push_back(targetKeypoints[good_matches[j].trainIdx].pt);
        }

        Mat H = findHomography(points_sample, points_target, RANSAC);

        if (H.empty())
            continue;

        vector<Point2f> corners_sample = { {0, 0}, {static_cast<float>(images[i].cols), 0},
                                          {static_cast<float>(images[i].cols), static_cast<float>(images[i].rows)}, {0, static_cast<float>(images[i].rows)} };
        vector<Point2f> corners_target;

        perspectiveTransform(corners_sample, corners_target, H);

        double area = contourArea(corners_target);
        if (area < 1000) continue;

        for (int j = 0; j < 4; j++) {
            line(imgMatches, corners_target[j], corners_target[(j + 1) % 4], Scalar(255, 0, 0), 3);
        }

        Point2f center(0, 0);
        for (const auto& pt : corners_target)
            center += pt;
        center *= (1.0 / corners_target.size());

        int baseline = 0;
        Size textSize = getTextSize(names[i], FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
        Point2f text_pos = center - Point2f(textSize.width / 2.0, textSize.height / 2.0);
        putText(imgMatches, names[i], text_pos, FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 215, 255), 2);
    }

    return imgMatches;
}

int main() {
    setlocale(LC_ALL, "ru");
    vector<string> names;
    vector<Mat> images;
    vector<vector<KeyPoint>> keypoints;
    vector<Mat> descriptors;

    Ptr<SIFT> sift = SIFT::create();

    string path("cards");
    for (auto& p : fs::recursive_directory_iterator(path)) {
        if (p.path().extension() == ".png") {
            if (!parseImage(p.path(), sift, images, keypoints, descriptors, names)) {
                continue;
            }
        }
    }

    Mat cards = imread("cards.png");
    if (cards.empty()) {
        cout << "Ошибка" << endl;
        return -1;
    }

    Mat imgMatches = matchImages(cards, images, keypoints, descriptors, names, sift);

    imshow("Обнаруженные карты", imgMatches);
    imwrite("Обнаруженные карты.png", imgMatches);
    while ((waitKey() & 0xEFFFFF) != 27);

    return 0;
}
