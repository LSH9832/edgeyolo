#ifndef TRT_H
#define TRT_H

#include <iostream>
#include <string.h>
#include <fstream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include "image_utils/logging.h"
#include "image_utils/detect_process.h"
#include "yaml-cpp/yaml.h"

using namespace nvinfer1;


struct CFG_MSG {
    std::string INPUT_NAME;     // input_0
    std::string OUTPUT_NAME;    // output_0
    cv::Size INPUT_SIZE;   // = cv::Size(384, 672);
    std::vector<std::string> NAMES;      
    bool NORMALIZE;        // 0
    bool loaded = false;
};

class yoloNet {

    int num_classes=0;
    int output_num_array=0;

    double conf_thres = 0.25;
    double nms_thres = 0.45;
    long long output_size = 1;
    cv::Size image_size = cv::Size(640, 640);
    size_t size{0};

    int num_names = 0;

    IRuntime* runtime;
    IExecutionContext* context;
    ICudaEngine* engine;

    int inputIndex;
    int outputIndex;
    int batch=1;

    bool first=true;
    bool cuda_occupyed=false;
    bool normalize=false;

    std::string input_name = "input_0";
    std::string output_name = "output_0";
    cudaStream_t stream;
    void* buffers[2];
    float* prob;

public:
    yoloNet();
    CFG_MSG cfg;

    bool load_engine(std::string cfg_file);
    bool engine_loaded=false;

    void set_conf_threshold(float thres);
    void set_nms_threshold(float thres);
    void set_input_name(std::string name);
    void set_output_name(std::string name);
    void release();

    std::vector<detect::Object> infer(cv::Mat image);

};

#endif
#define TRT_H
