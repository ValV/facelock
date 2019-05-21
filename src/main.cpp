#include "opencv2/core/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/face/facemark.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/dnn/dnn.hpp"

#include <sys/stat.h>
#include <iostream>
//#include <string>

using namespace cv;
using namespace cv::dnn;
using namespace cv::face;
using namespace std;

inline bool exists(const std::string &path) {
  struct stat buffer;
  return (stat (path.c_str(), &buffer) == 0);
};

const char *keys =
  "{ help h       |     | Print help message. }"
  "{ model m      |opencv_face_detector.pbtxt|"
  "                       Path to Tensorflow model (.pbtxt). }"
  "{ weights w    |opencv_face_detector_uint8.pb|"
  "                       Path to Tensorflow model weights (.pb). }"
  "{ landmarks l  |face_landmark_model.dat|"
  "                       Path to Kazemi landmark weights (.dat). }"
  "{ width x      | 224 | Affine map window width (x axis) from 32px. }"
  "{ height y     | 224 | Affine map window height (y axis) from 32px. }"
  "{ confidence c | 0.7 | Initial confidence for face detection. }"
  "{ showpts s    |false| Show landmark points + auxiliary points. }"
  ;

int main(int argc, char *argv[]) {
  // Configure program command line options
  CommandLineParser parser(argc, argv, keys);
  parser.about("Face detection OpenCV application");
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }

  try {
    CV_Assert(parser.get<int>("width") > 31);
    CV_Assert(parser.get<int>("height") > 31);
    CV_Assert(parser.get<float>("confidence") > 0.1
        && parser.get<float>("confidence") <= 1.0);
    CV_Assert(exists(parser.get<string>("model")));
    CV_Assert(exists(parser.get<string>("weights")));
    CV_Assert(exists(parser.get<string>("landmarks")));
  } catch (...) {
    parser.printMessage();
    cout << "Error: wrong command line or missing files!" << endl;
    return 1;
  }

  bool showpts = parser.get<bool>("showpts");
  int affineWidth = parser.get<int>("width");
  int affineHeight = parser.get<int>("height");
  float confidenceThreshold = parser.get<float>("confidence");
  float eyeDesired[] = {0.35, 0.35}; // desired left eye x, y shift

  // Open video capture device
  VideoCapture cap(0); // 0 - default video capture device
  if (!cap.isOpened()) {
    cerr << "Capture device ID 0 cannot be opened." << endl;
    return 2;
  }

  // Create SSD network from Tensorflow model
  Net ssdNet = readNetFromTensorflow(parser.get<string>("weights"),
      parser.get<string>("model"));

  // Create face landmark predictor
  Ptr<Facemark> facemark = createFacemarkKazemi();
  facemark->loadModel(parser.get<string>("landmarks"));

  // Process video input per frame
  Mat frame;
  while (true) {
    // Get current frame, convert it and detect a face
    cap >> frame;
    Mat original = frame.clone();
    // Prepare neural network (blob of certain size and color)
    Mat inputBlob = blobFromImage(original, 1.0,
        Size(300, 300), Scalar(103.93, 116.77, 123.68),
        true, false);
    ssdNet.setInput(inputBlob, "data");
    // Run network and get results
    Mat detection = ssdNet.forward("detection_out");
    Mat detectionMat(detection.size[2], detection.size[3],
        CV_32F, detection.ptr<float>());
    // Process results and get faces regions
    vector<Rect_<int>> faces;
    for (int i = 0; i < detectionMat.rows; i ++) {
      float confidence = detectionMat.at<float>(i, 2);
      if (confidence > confidenceThreshold) {
        int x1 = static_cast<int>(detectionMat.at<float>(i, 3)
            * original.cols);
        int y1 = static_cast<int>(detectionMat.at<float>(i, 4)
            * original.rows);
        int x2 = static_cast<int>(detectionMat.at<float>(i, 5)
            * original.cols);
        int y2 = static_cast<int>(detectionMat.at<float>(i, 6)
            * original.rows);
        faces.push_back(Rect(x1, y1, x2 - x1, y2 - y1));
      }
    }

    // Now faces variable holds rectangles for all detected faces
    // in the current frame. Annotate a face in the frame
    if (faces.size() == 1) {
      Rect face_rect = faces[0];
      vector<vector<Point2f>> shapes;
      if (facemark->fit(original, faces, shapes)) {
        // Calculate a center of each eye and a center between eyes
        Point2f eyeLeft = (shapes[0][42] + shapes[0][45]) / 2;
        Point2f eyeRight = (shapes[0][36] + shapes[0][39]) / 2;
        Point2f eyesCenter = (eyeLeft + eyeRight) / 2;
        // Calculate shifts, scale, and rotation angle
        float dX = eyeRight.x - eyeLeft.x;
        float dY = eyeRight.y - eyeLeft.y;
        float scale = (1.0 - eyeDesired[0] - eyeDesired[0]) * affineWidth
          / (sqrt(pow(dX, 2) + pow(dY, 2)));
        float angle = atan2(dY, dX) * 180 / M_PI - 180;
        // Get affine rotation matrix
        Mat M = getRotationMatrix2D(eyesCenter, angle, scale);
        // Adjust eyes' center target point
        float tX = affineWidth * 0.5;
        float tY = affineHeight * eyeDesired[1];
        M.at<double>(0, 2) += (tX - eyesCenter.x);
        M.at<double>(1, 2) += (tY - eyesCenter.y);
        // Create affine transformation (map)
        Mat affine(affineWidth, affineHeight, CV_32F);
        warpAffine(original, affine, M, Size(affineWidth, affineHeight));
        // Draw face landmarks
        if (showpts) {
          // Left eye outer/inner points
          circle(original, shapes[0][36], 2, CV_RGB(30, 144, 255), FILLED);
          circle(original, shapes[0][39], 2, CV_RGB(30, 144, 255), FILLED);
          // Right eye inner/outer points
          circle(original, shapes[0][42], 2, CV_RGB(30, 144, 255), FILLED);
          circle(original, shapes[0][45], 2, CV_RGB(30, 144, 255), FILLED);
          // Auxiliary eye points
          circle(original, eyesCenter, 3, CV_RGB(221, 160, 221), FILLED);
          circle(original, eyeRight, 3, CV_RGB(240, 230, 140), FILLED);
          circle(original, eyeLeft, 3, CV_RGB(240, 230, 140), FILLED);
          line(original, eyeRight, eyeLeft, CV_RGB(221, 160, 221),
              1, LINE_AA);
        }
        // Highlight a face with a rectangle
        rectangle(original, face_rect, CV_RGB(250, 128, 114));
        // Draw border around affine map
        rectangle(affine, Rect(0, 0, affine.cols, affine.rows),
            CV_RGB(112, 128, 144), 2);
        // Draw affine map on the frame
        affine.copyTo(original(Rect(original.cols - affineWidth,
              original.rows - affineHeight, affineWidth, affineHeight)));
      }
      // Display the frame with face detected (and affine map applied)
      imshow("facelock", (original));
    }

    // Wait and catch keypress (ESC)
    char key = (char) waitKey(250);
    if (key == 27) break;
  }

  destroyAllWindows();
  return 0;
}

// vim: se et ts=2 sw=2 number:
