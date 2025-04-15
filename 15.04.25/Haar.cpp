#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    CascadeClassifier face_cascade, eyes_cascade, smile_cascade;

    if (!face_cascade.load("haarcascade_frontalface_default.xml") ||
        !eyes_cascade.load("haarcascade_eye.xml") ||
        !smile_cascade.load("haarcascade_smile.xml")) {
        cout << "Ошибка загрузки каскадов" << endl;
        return -1;
    }

    VideoCapture cap("ZUA.mp4");
    if (!cap.isOpened()) {
        cout << "Не удалось открыть видео" << endl;
        return -1;
    }

    int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(CAP_PROP_FPS);

    VideoWriter video("result.avi",
        VideoWriter::fourcc('M', 'J', 'P', 'G'),
        fps,
        Size(frame_width, frame_height));
    if (!video.isOpened()) {
        cout << "Не удалось создать видеофайл для сохранения" << endl;
        return -1;
    }

    Mat frame, gray;
    while (cap.read(frame)) {
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);

        vector<Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.3, 5);

        for (const auto& face : faces) {
            rectangle(frame, face, Scalar(255, 0, 0), 2);
            Mat faceROI = gray(face);
            Mat faceColor = frame(face);

            vector<Rect> eyes;
            eyes_cascade.detectMultiScale(faceROI, eyes, 1.08, 20, 0, Size(55, 50));
            for (const auto& eye : eyes) {
                Rect eyeRect(eye.x + face.x, eye.y + face.y, eye.width, eye.height);
                rectangle(frame, eyeRect, Scalar(0, 255, 0), 2);
            }

            vector<Rect> smiles;
            smile_cascade.detectMultiScale(faceROI, smiles, 1.2, 50);
            for (const auto& smile : smiles) {
                Rect smileRect(smile.x + face.x, smile.y + face.y, smile.width, smile.height);
                rectangle(frame, smileRect, Scalar(0, 255, 255), 2);
            }
        }

        video.write(frame);

        imshow("Распознавание", frame);
        if (waitKey(10) == 27) break;
    }

    cap.release();
    video.release();
    destroyAllWindows();
    return 0;
}