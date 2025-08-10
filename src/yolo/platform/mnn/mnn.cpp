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


template <class T>
size_t indexOf(std::vector<T> lists, T elem)
{
    for (size_t i=0; i < lists.size(); i++)
    {
        if (lists[i] == elem) return i;
    }
    return -1;
}

template <class T>
void showShape(std::vector<T> shape)
{
    std::ostringstream oss;
    oss << "[";
    for (int i=0;i<shape.size();i++)
    {
        if (i) oss << ", ";
        oss << shape[i];
    }
    oss << "]";
    std::cout << oss.str() << std::endl;
}

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
    // MNN::Tensor *outputTensor = nullptr;

    std::vector<MNN::Tensor *> outputTensors, retTensors;

    MNN::Tensor* nchwTensor = nullptr;
    // MNN::Tensor* retTensor = nullptr;

    float confThres=0.25, nmsThres=0.45;

    bool regExp=true;
    bool regMul=false;

    bool anchorBased = false;
    std::vector<std::vector<std::vector<float>>> anchors;
    std::vector<int> gridsX, gridsY;

    int numAnchors = 1;      // anchor free = 1
    bool needDecoder=true;   // num of outputs > 1
    bool splitHead=true;     // num of outputs == num strides * 3

    bool first=true;
    bool init=false;
    bool obj_conf_enabled=true;
    bool deviceOccupied=false;
    long long outputSize = 0;
    int batch = 1;
    int numArrays=0;
    int numClasses=0;
    int lengthArray=0;


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
        else if (type == "vulkan")
        {
            impl_->config.type = MNN_FORWARD_VULKAN;
        }
    }
    else if (key == "anchors")
    {
        /* code */
        std::cout << "loading anchors" << std::endl;
        impl_->anchors.clear();
        impl_->anchorBased = false;
        for (pystring thisStrideAnchors: pystring(value).split(";"))
        {
            std::vector<std::vector<float>> oneStrideAnchors;
            for(pystring thisAnchor: thisStrideAnchors.split(","))
            {
                std::vector<float> oneAnchor;
                std::vector<pystring> oneAnchorStr = thisAnchor.split(" ");
                oneAnchor.push_back(oneAnchorStr[0]);
                oneAnchor.push_back(oneAnchorStr[1]);
                oneStrideAnchors.push_back(oneAnchor);
            }
            impl_->numAnchors = oneStrideAnchors.size();
            impl_->anchors.push_back(oneStrideAnchors);
        }
        if (impl_->anchors.size()) impl_->anchorBased = true;
        else impl_->numAnchors = 1;
    }
    else if (key == "version")   // yolo v ?
    {
        pystring type = pystring(value).lower();
        if (type == "yolov3" || type == "edgeyolo" || type == "yolox")
        {
            impl_->regExp = true;
            impl_->regMul = false;
            impl_->obj_conf_enabled = true;
        }
        else if (type == "yolov5" || type == "yolov7" || type == "yolov8" || type == "yolov9" || type == "yolov10")
        {
            impl_->regExp = false;
            impl_->regMul = true;
            impl_->obj_conf_enabled = true;
        }
        else if (type == "yolov6")
        {
            impl_->obj_conf_enabled = false;
        }
        else
        {
            std::cout << "unrecognized yolo type '" << value << "'" << std::endl;
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
    impl_->config.backendConfig->precision = MNN::BackendConfig::Precision_Low;
    impl_->config.backendConfig->power = MNN::BackendConfig::Power_Normal;
    impl_->config.backendConfig->memory = MNN::BackendConfig::Memory_Normal;
    impl_->net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(modelFile_.c_str()));
    impl_->session = impl_->net->createSession(impl_->config);

    //获取输入tensor
    impl_->inputTensor = impl_->net->getSessionInput(impl_->session, NULL);
    assert(impl_->inputTensor->shape().size() == 4);    // n, c, h, w
    impl_->batch = impl_->inputTensor->shape().at(0);
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

    //获取输出tensors
    impl_->outputTensors.clear();
    auto outputs = impl_->net->getSessionOutputAll(impl_->session);

    std::vector<std::string> allOutputNames;
    for(auto out = outputs.begin();out != outputs.end();++out)
    {
        allOutputNames.push_back(out->first);
    }
    
    for (std::string outName: outputNames_)
    {
        
        if(indexOf(allOutputNames, outName) < 0)
        {
            std::cerr << "[E] can not find output name '" << outName << "'!";
            return false;
        }

        impl_->outputTensors.push_back(outputs[outName]);
        impl_->retTensors.push_back(new MNN::Tensor(outputs[outName], MNN::Tensor::CAFFE));
        std::cout << outName << ": "; 
        showShape(outputs[outName]->shape());
    }
    impl_->needDecoder = outputNames_.size() > 1;

    if (impl_->needDecoder)
    {
        if (outputNames_.size() == strides_.size() * 3)   // 9 in general
        {
            impl_->splitHead = true;
            impl_->lengthArray = impl_->outputTensors[strides_.size() * 2]->shape()[1];
            impl_->numClasses = impl_->lengthArray;
        }
        else if (outputNames_.size() == strides_.size()) // 3 in general
        {
            impl_->splitHead = false;
            impl_->lengthArray = impl_->outputTensors[0]->shape()[1];
            if (impl_->anchors.size()) 
            {
                impl_->lengthArray /= impl_->anchors[0].size();
            }
            impl_->numClasses = impl_->lengthArray - (impl_->obj_conf_enabled?5:4);
        }
        else 
        {
            std::cerr << "[E] num outs is " << outputNames_.size() << ", which is not supported." << std::endl;
            exit(-1);
        }
    }
    else
    {
        // impl_->outputTensor = impl_->net->getSessionOutput(impl_->session, NULL);
        impl_->numArrays = impl_->outputTensors[0]->shape().at(1);
        impl_->lengthArray = impl_->outputTensors[0]->shape().at(2);
        impl_->numClasses = impl_->lengthArray - (impl_->obj_conf_enabled?5:4);
        assert(impl_->outputTensors[0]->shape().size() == 3);   // batch, num_dets, array
        // impl_->retTensor = new MNN::Tensor(impl_->outputTensors[0], MNN::Tensor::CAFFE);
    }


    if (impl_->needDecoder) {
        impl_->numArrays = 0;
        for(int i=0; i<outputNames_.size(); i++) 
        {
            if (impl_->splitHead && i < strides_.size() * 2) continue;
            int gx = impl_->outputTensors[i]->shape()[3];
            int gy = impl_->outputTensors[i]->shape()[2];
            impl_->gridsX.push_back(gx);
            impl_->gridsY.push_back(gy);
            impl_->numArrays += impl_->numAnchors * gx * gy;
        }
    }

    impl_->outputSize = impl_->batch * impl_->numArrays * impl_->lengthArray;
    
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
    double t0 = pytime::time();
    if (data != nullptr)
    {
        if (impl_->nchwTensor->host<void>() != data)
        {
            memcpy(impl_->nchwTensor->host<float>(), data, 3 * imgH_ * imgW_ * sizeof(float));
        }
        // else
        // {
        //     std::cout << "no need to copy data" << std::endl;
        // }
    }
    
    double dt = pytime::time() - t0;
    // std::cout << "  copy1: " << dt << "ms,";
    t0 = pytime::time();

    impl_->inputTensor->copyFromHostTensor(impl_->nchwTensor);

    dt = pytime::time() - t0;
    // std::cout << " copy2: " << dt * 1000 << "ms";
    t0 = pytime::time();

    impl_->net->runSession(impl_->session);

    dt = pytime::time() - t0;
    // std::cout << " infer: " << dt * 1000 << "ms";
    t0 = pytime::time();
    // impl_->outputTensor->copyToHostTensor(impl_->retTensor);

    for(size_t i=0;i<impl_->retTensors.size();++i)
    {
        impl_->outputTensors[i]->copyToHostTensor(impl_->retTensors[i]);
    }

    dt = pytime::time() - t0;
    // std::cout << " copy3: " << dt * 1000 << "ms";

    // t0 = pytime::time();
    // ----------- post process -------------
    std::vector<std::vector<float>> results;
    if (impl_->needDecoder)   // num outputs > 1
    {
        int numStrides = strides_.size();
        if(impl_->splitHead)
        {
            std::vector<std::vector<float>> nowAnchor;
            for(int i=0;i<strides_.size();i++)
            {
                if (impl_->anchorBased)
                {
                    nowAnchor = impl_->anchors[i];
                }
                else
                {
                    nowAnchor = {{1.0f, 1.0f}};
                }
                
                std::vector<std::vector<float>> result = decodeOutputs(
                    impl_->retTensors[i + numStrides * 0]->host<void>(),
                    impl_->retTensors[i + numStrides * 1]->host<void>(),
                    impl_->retTensors[i + numStrides * 2]->host<void>(),
                    impl_->numClasses, true, true, impl_->obj_conf_enabled, 
                    impl_->gridsX[i], impl_->gridsY[i], strides_[i], nowAnchor,
                    0, 0, 0, 0, 0, 0,
                    impl_->regExp, impl_->regMul, 0, impl_->confThres
                );

                // std::cout << "result len: " << result.size() << std::endl;

                results.insert(results.end(), result.begin(), result.end());
            }
            float* preds_ = (float*)preds;
            preds_[0] = results.size();
            int idx = 1;
            for (auto this_result: results)
            {
                for(int i=0;i<6;i++)
                {
                    preds_[idx + i] = this_result[i];
                }
                idx += 6;
            }
        }
        else
        {
            std::vector<std::vector<float>> nowAnchor;
            
            for(int i=0;i<strides_.size();i++)
            {
                if (impl_->anchorBased)
                {
                    nowAnchor = impl_->anchors[i];
                }
                else
                {
                    nowAnchor = {{(float)strides_[i], (float)strides_[i]}};
                }

                std::vector<std::vector<float>> result = decodeOutputs(
                    impl_->retTensors[i]->host<void>(),
                    nullptr, 
                    nullptr,
                    impl_->numClasses, false, true, impl_->obj_conf_enabled, 
                    impl_->gridsX[i], impl_->gridsY[i], 
                    strides_[i], nowAnchor,
                    0, 0, 0, 0, 0, 0,
                    impl_->regExp, impl_->regMul, 0, impl_->confThres
                );

                results.insert(results.end(), result.begin(), result.end());
            }
        }
    }
    else
    {
        generate_yolo_proposals(
            impl_->numArrays, 
            (float*)impl_->retTensors[0]->host<void>(),
            impl_->confThres, results, impl_->numClasses
        );
        // memcpy(preds, impl_->outputs[0].buf, impl_->outputSize * sizeof(float));
    }
    std::vector<int> picked;

    // generate_yolo_proposals(impl_->numArrays, impl_->retTensor->host<float>(), impl_->confThres, results, impl_->numClasses);
    qsort_descent_inplace(results);
    nms_sorted_bboxes(results, picked, impl_->nmsThres);

    float* preds_ = (float*)preds;
    preds_[0] = picked.size();
    int idx = 1;
    // std::cout << results.size() << std::endl;
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

bool YOLO::inputDataReachable()
{
    if (impl_ == nullptr)
    {
        std::cerr << "[E] yolo not init!" << std::endl;
        return false;
    }
    else if (!impl_->init)
    {
        std::cerr << "[E] yolo not init!" << std::endl;
        return false;
    }
    return true;
}

void* YOLO::getInputData() 
{
    if (!inputDataReachable()) return nullptr;
    return impl_->nchwTensor->host<void>();
}


YOLO::~YOLO()
{
    delete impl_->backendConfig;
    // delete impl_->session;
    delete impl_->inputTensor;
    // delete impl_->outputTensor;
    for (int i=0;i<impl_->outputTensors.size();i++)
    {
        delete impl_->outputTensors[i];
        delete impl_->retTensors[i];
    }
    
    delete impl_->nchwTensor;
    // delete impl_->retTensor;
    impl_->backendConfig = nullptr;
    // impl_->session = nullptr;
    impl_->inputTensor = nullptr;
    impl_->outputTensors.clear();
    impl_->retTensors.clear();
    impl_->nchwTensor = nullptr;
    // impl_->retTensor = nullptr;
}


#endif
