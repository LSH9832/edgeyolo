#ifndef YOLO_MNN_CPP
#define YOLO_MNN_CPP

#include <iostream>
#include <string.h>
#include <fstream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>

#include "../../yolo.hpp"
#include "../../str.h"
#include "../../datetime.h"
#include "../common.hpp"

// #include "../FastMemcpy/FastMemcpy.h" // 不好用

#include "MNN/MNNDefine.h"
#include "MNN/expr/Expr.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN/AutoTime.hpp"
#include "MNN/Interpreter.hpp"

/* ----------------------- implimentation ----------------------- */

void platform(char* p)
{
    std::string p_ = "MNN";
    for (int i=0;i<p_.size();i++)
    {
        p[i] = p_[i];
    }
}


struct YOLO::Impl
{
    std::shared_ptr<MNN::Interpreter> net; //模型翻译创建
    MNN::ScheduleConfig config;            //计划配置
    MNN::BackendConfig* backendConfig = nullptr;
    MNN::Session *session = nullptr;
    MNN::Tensor *inputTensor = nullptr;
    MNN::Tensor *outputTensor = nullptr;

    MNN::Tensor* nchwTensor = nullptr;
    MNN::Tensor* retTensor = nullptr;

    float confThres=0.25, nmsThres=0.45;

    bool first=true;
    bool init=false;
    bool obj_conf_enabled=true;
    bool deviceOccupied=false;
    long long outputSize = 0;
    int batch = 1;
    int numArrays=0;
    int numClasses=0;


    // debug 
    double t_sum=0.;
    int count = 0;
};


YOLO::YOLO(
    std::string modelFile, std::string inputName, std::vector<std::string> outputNames,
    int imgW, int imgH, std::vector<int> strides, int device
)
{
    modelFile_ = modelFile;
    inputName_ = inputName;
    outputNames_ = outputNames;    // mnn同理，不需要在网络外计算grid，放到网络中即可，只输出一个结果
    imgW_ = imgW;
    imgH_ = imgH;
    strides_ = strides;
    if (impl_ == nullptr)
    {
        impl_ = new Impl();
    }
}


void YOLO::set(std::string key, std::string value)
{
    // std::cout << key << "," << value << std::endl;
    if (key == "threads")
    {
        // std::cout << 1 << "," << (int)pystring(value) << std::endl;
        impl_->config.numThread = (int)pystring(value);
        // std::cout << 2 << std::endl;
    }
    else if (key == "type")
    {
        pystring type = pystring(value).lower();
        if (type == "cpu")
        {
            impl_->config.type = MNN_FORWARD_CPU;
        }
        else if (type == "cuda")
        {
            impl_->config.type = MNN_FORWARD_CUDA;
        }
        else if (type == "all")
        {
            impl_->config.type = MNN_FORWARD_ALL;
        }
        else if (type == "auto")
        {
            impl_->config.type = MNN_FORWARD_AUTO;
        }
        else if (type == "opencl")
        {
            impl_->config.type = MNN_FORWARD_OPENCL;
        }
        else if (type == "opengl")
        {
            impl_->config.type = MNN_FORWARD_OPENGL;
        }
        else if (type == "metal")
        {
            impl_->config.type = MNN_FORWARD_METAL;
        }
        else if (type == "nn")
        {
            impl_->config.type = MNN_FORWARD_NN;
        }
        else if (type == "VULKAN")
        {
            impl_->config.type = MNN_FORWARD_VULKAN;
        }
    }
    else if (key == "conf_threshold")
    {
        impl_->confThres = pystring(value);
    }
    else if (key == "nms_threshold")
    {
        impl_->nmsThres = pystring(value);
    }
    else
    {
        std::cout << "unused setting -> " << key << ": " << value << std::endl;
    }
}


bool YOLO::init()
{
    if (impl_ != nullptr)
    {
        if (impl_->init) return true;
    }
    else
    {
        impl_ = new Impl();
    }

    if (impl_->backendConfig == nullptr)
    {
        impl_->backendConfig = new MNN::BackendConfig();
        impl_->config.backendConfig = impl_->backendConfig;
    }
    impl_->net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(modelFile_.c_str()));
    impl_->config.backendConfig->precision = MNN::BackendConfig::Precision_Low;
    impl_->config.backendConfig->power = MNN::BackendConfig::Power_Normal;
    impl_->config.backendConfig->memory = MNN::BackendConfig::Memory_Normal;
    impl_->session = impl_->net->createSession(impl_->config);

    //获取输入输出tensor
    impl_->inputTensor = impl_->net->getSessionInput(impl_->session, NULL);
    impl_->outputTensor = impl_->net->getSessionOutput(impl_->session, NULL);

    impl_->numArrays = impl_->outputTensor->shape().at(1);
    impl_->numClasses = impl_->outputTensor->shape().at(2) - (impl_->obj_conf_enabled?5:4);

    assert(impl_->inputTensor->shape().size() == 4);    // n, c, h, w
    assert(impl_->outputTensor->shape().size() == 3);   // batch, num_dets, array

    impl_->batch = impl_->inputTensor->shape().at(0);
    impl_->outputSize = impl_->batch * impl_->numArrays * impl_->outputTensor->shape().at(2);


    if (imgH_ != impl_->inputTensor->shape().at(2))
    {
        std::cout << "[W] wrong settings of input image height(" << imgH_ << "), which should be " << impl_->inputTensor->shape().at(2) << std::endl;
        imgH_ = impl_->inputTensor->shape().at(2);
    }
    if (imgW_ != impl_->inputTensor->shape().at(3))
    {
        std::cout << "[W] wrong settings of input image width(" << imgW_ << "), which should be " << impl_->inputTensor->shape().at(3) << std::endl;
        imgW_ = impl_->inputTensor->shape().at(3);
    }
    impl_->nchwTensor = new MNN::Tensor(impl_->inputTensor, MNN::Tensor::CAFFE);
    impl_->retTensor = new MNN::Tensor(impl_->outputTensor, MNN::Tensor::CAFFE);
    impl_->init = true;
    return true;

}


bool YOLO::isInit()
{
    if (impl_ != nullptr)
    {
        return impl_->init;
    }
    return false;
}


int YOLO::getNumClasses()
{
    return impl_->numClasses;
}


int YOLO::getNumArrays()
{
    return impl_->numArrays;
}


void YOLO::inference(void* data, void* preds, float scale)
{
    // *impl_->nchwTensor->host<float>() = *data;
    // memcpy: 0.45244 ms
    // memcpy_fast: 0.515268 ms 
    // double t0 = pytime::time();
    memcpy(impl_->nchwTensor->host<float>(), data, 3 * imgH_ * imgW_ * sizeof(float));
    // double dt = pytime::time() - t0;
    // impl_->t_sum += dt;
    // impl_->count += 1;
    // std::cout << "  copy1_avg(memcopy_fast): " << impl_->t_sum * 1000 / impl_->count << "ms,";

    // t0 = pytime::time();
    impl_->inputTensor->copyFromHostTensor(impl_->nchwTensor);
    // dt = pytime::time() - t0;
    // std::cout << " copy2: " << dt * 1000 << "ms";

    // t0 = pytime::time();
    impl_->net->runSession(impl_->session);
    // dt = pytime::time() - t0;
    // std::cout << " infer: " << dt * 1000 << "ms";
    // pytime::sleep(1);

    // t0 = pytime::time();
    impl_->outputTensor->copyToHostTensor(impl_->retTensor);
    // dt = pytime::time() - t0;
    // std::cout << " copy3: " << dt * 1000 << "ms";

    // t0 = pytime::time();
    std::vector<std::vector<float>> results;
    std::vector<int> picked;

    generate_yolo_proposals(impl_->numArrays, impl_->retTensor->host<float>(), impl_->confThres, results, impl_->numClasses);
    qsort_descent_inplace(results);
    nms_sorted_bboxes(results, picked, impl_->nmsThres);

    float* preds_ = (float*)preds;
    preds_[0] = picked.size();
    int idx = 1;
    for (int p: picked)
    {
        std::vector<float>& this_result = results[p];
        preds_[idx] = this_result[0] / scale;
        preds_[idx+1] = this_result[1] / scale;
        preds_[idx+2] = preds_[idx] + this_result[2] / scale;
        preds_[idx+3] = preds_[idx+1] + this_result[3] / scale;
        preds_[idx+4] = this_result[4];
        preds_[idx+5] = this_result[5];
        idx += 6;
    }

    // memcpy(preds, impl_->retTensor->host<float>(), impl_->outputSize * sizeof(float));
    // dt = pytime::time() - t0;
    // std::cout << " copy4: " << dt * 1000 << "ms" << std::endl;
}


YOLO::~YOLO()
{
    delete impl_->backendConfig;
    // delete impl_->session;
    delete impl_->inputTensor;
    delete impl_->outputTensor;
    delete impl_->nchwTensor;
    delete impl_->retTensor;
    impl_->backendConfig = nullptr;
    // impl_->session = nullptr;
    impl_->inputTensor = nullptr;
    impl_->outputTensor = nullptr;
    impl_->nchwTensor = nullptr;
    impl_->retTensor = nullptr;
}


#endif