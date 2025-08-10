#ifndef YOLO_DETECT_HPP
#define YOLO_DETECT_HPP

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>


enum AccPlatform
{
    ACC_PLAT_AUTO,
    ACC_PLAT_MNN,
    ACC_PLAT_TENSORRT,
    ACC_PLAT_RKNN,
    ACC_PLAT_HBDNN,
    ACC_PLAT_OPENVINO,
    ACC_PLAT_ASCEND,
    ACC_PLAT_ONNX,
};


class Detector
{
public:

    Detector() {}

    ~Detector();

    bool init(std::string configFile, AccPlatform platform = ACC_PLAT_AUTO);

    std::string platform();

    void preProcess(const cv::Mat& image, cv::Mat& resized, float& ratio);

    void infer(cv::Mat& image, float ratio, std::vector<std::vector<float>>& prediction);

    void detect(const cv::Mat& image, std::vector<std::vector<float>>& prediction, cv::Size* oriSz=nullptr);

    void set(std::string key, std::string value);

    std::vector<std::string> getNames();

    cv::Mat draw(cv::Mat& src, std::vector<std::vector<float>>& prediction, bool draw_label=true, int thickness=20);

    struct Impl;
private:
    Impl* impl_=nullptr;
};


#endif