#ifndef YOLO_TENSORRT_CPP
#define YOLO_TENSORRT_CPP

#include <iostream>
#include <string.h>
#include <fstream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <cuda_runtime_api.h>
#include <NvInfer.h>

#include "../../yolo.hpp"
#include "../common.hpp"
#include "../../str.h"
#include "./trt_logging.h"


#define CHECK(status) \
    do\
    {\
        int ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)


using namespace nvinfer1;


static nvLogger gLogger(ILogger::Severity::kERROR);   /* ILogger::Severity::kWARNING */

/* ----------------------- implimentation ----------------------- */

void platform(char* p)
{
    std::string p_ = "TensorRT";
    for (int i=0;i<p_.size();i++)
    {
        p[i] = p_[i];
    }
}

struct YOLO::Impl
{
    IRuntime* runtime;
    IExecutionContext* context;
    ICudaEngine* engine;
    cudaStream_t stream;
    void* buffers[2];

    float confThres=0.25, nmsThres=0.45;

    bool first=true;
    bool init=false;
    bool obj_conf_enabled=true;
    bool cudaOccupied=false;
    long long outputSize = 0;
    int batch = 1;
    int numArrays=0;
    int numClasses=0;
    int inputIndex=0, outputIndex=0;
};


YOLO::YOLO(
    std::string modelFile, std::string inputName, std::vector<std::string> outputNames,
    int imgW, int imgH, std::vector<int> strides, int device
)
{
    modelFile_ = modelFile;
    inputName_ = inputName;
    outputNames_ = outputNames;    // tensorrt不需要在网络外计算grid，放到网络中即可，只输出一个结果
    imgW_ = imgW;
    imgH_ = imgH;
    strides_ = strides;
    cudaSetDevice(device);
    init();
}

void YOLO::set(std::string key, std::string value)
{
    if (key == "conf_threshold")
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

    // read model
    char *trtModelStream{nullptr};
    std::ifstream file(modelFile_, std::ios::binary);
    size_t size{0};
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    else
    {
        std::cerr << "can not load from file: " << modelFile_ << std::endl;
        exit(-1);
    }

    impl_->runtime = createInferRuntime(gLogger);
    assert(impl_->runtime != nullptr);
    impl_->engine = impl_->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(impl_->engine != nullptr);
    impl_->context = impl_->engine->createExecutionContext();
    assert(impl_->context != nullptr);
    delete[] trtModelStream;

    // std::cout << inputName_ << ", " << outputNames_[0] << std::endl;

    assert(impl_->engine->getNbBindings() == 2);
    impl_->inputIndex = impl_->engine->getBindingIndex(inputName_.c_str());
    assert(impl_->engine->getBindingDataType(impl_->inputIndex) == nvinfer1::DataType::kFLOAT);
    impl_->outputIndex = impl_->engine->getBindingIndex(outputNames_[0].c_str());
    assert(impl_->engine->getBindingDataType(impl_->outputIndex) == nvinfer1::DataType::kFLOAT);

#ifdef TRT7
    auto out_dims = impl_->engine->getBindingDimensions(inputIndex);
#else
    auto out_dims = impl_->engine->getTensorShape(outputNames_[0].c_str());
#endif

    impl_->outputSize = 1;
    for(int j=0;j<out_dims.nbDims;j++) {

        if (j==0) {
            int batch = out_dims.d[j];
            if (batch == -1){
                batch = impl_->engine->getMaxBatchSize();
            }
            if (impl_->batch > batch)
                impl_->batch = batch;
            impl_->outputSize *= impl_->batch;
            out_dims.d[j] = impl_->batch;
            // cout<<"batch size of this engine file: "<<impl_->batch<<endl;

        }
        else impl_->outputSize *= out_dims.d[j];

        if (j == out_dims.nbDims - 2)
        {
            impl_->numArrays = out_dims.d[j];
        }
        if (j == out_dims.nbDims - 1)
        {
            impl_->numClasses = out_dims.d[j] - (impl_->obj_conf_enabled?5:4);
        }

        // std::cout << out_dims.d[j] << ", ";

    }
    // std::cout << std::endl;
    // std::cout << impl_->batch << std::endl;

#ifdef TRT7
    impl_->context->setBindingDimensions(impl_->inputIndex, out_dims);
#elif defined(TRT8)
    impl_->context->setBindingDimensions(impl_->inputIndex, out_dims);
#else
    
#endif
    // impl_->numArrays = countLengthArray(imgW_, imgH_);
    // impl_->numClasses = (impl_->outputSize / impl_->batch / impl_->numArrays) - (impl_->obj_conf_enabled?5:4);
    CHECK(cudaStreamCreate(&impl_->stream));

    // std::cout << "length array: " << impl_->numArrays << 
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
    // first init 
    if (impl_->first) {
        // Create GPU buffers on device
        CHECK(cudaMalloc(&impl_->buffers[impl_->inputIndex], 3 * imgH_ * imgW_ * sizeof(float)));
        CHECK(cudaMalloc(&impl_->buffers[impl_->outputIndex], impl_->outputSize * sizeof(float)));

#ifdef TRT7
#elif defined(TRT8)
#else
        impl_->context->setTensorAddress(inputName_.c_str(), impl_->buffers[impl_->inputIndex]);
        impl_->context->setTensorAddress(outputNames_[0].c_str(), impl_->buffers[impl_->outputIndex]);
#endif
        impl_->first = false;
    }

    // inference
    CHECK(cudaMemcpyAsync(impl_->buffers[impl_->inputIndex], data, 3 * imgH_ * imgW_ * sizeof(float), cudaMemcpyHostToDevice, impl_->stream));
    
#ifdef TRT7
    impl_->context->enqueue(1, impl_->buffers, impl_->stream, nullptr);    //TensorRT 7.X.X
#elif defined(TRT8)
    impl_->context->enqueueV2(impl_->buffers, impl_->stream, nullptr);     //TensorRT 8.X.X
#else
    impl_->context->enqueueV3(impl_->stream);
#endif
    CHECK(cudaMemcpyAsync(preds, impl_->buffers[impl_->outputIndex], impl_->outputSize * sizeof(float), cudaMemcpyDeviceToHost, impl_->stream));
    cudaStreamSynchronize(impl_->stream);
    impl_->cudaOccupied = true;

    std::vector<std::vector<float>> results;
    std::vector<int> picked;

    generate_yolo_proposals(impl_->numArrays, (float*)preds, impl_->confThres, results, impl_->numClasses);
    qsort_descent_inplace(results);
    nms_sorted_bboxes(results, picked, impl_->nmsThres);

    float* preds_ = (float*)preds;
    preds_[0] = picked.size();
    int idx = 1;
    scale = std::max(0.001f, scale);
    for (int p: picked)
    {
        std::vector<float>& this_result = results[p];

        preds_[idx] = this_result[0] / scale;
        preds_[idx+1] = this_result[1] / scale;
        preds_[idx+2] = preds_[idx] + this_result[2] / scale;
        preds_[idx+3] = preds_[idx+1] + this_result[3] / scale;
        preds_[idx+4] = this_result[4];
        preds_[idx+5] = this_result[5];

        // for(int i=0;i<6;i++)
        // {
        //     preds_[idx + i] = this_result[i];
        // }
        idx += 6;
    }

}

bool YOLO::inputDataReachable()
{
    return false;
}

void* YOLO::getInputData() 
{
    return nullptr;
}


YOLO::~YOLO()
{
    // Release stream and buffers
    if (isInit()){
        cudaStreamDestroy(impl_->stream);
        if(impl_->cudaOccupied) {
            CHECK(cudaFree(impl_->buffers[impl_->inputIndex]));
            CHECK(cudaFree(impl_->buffers[impl_->outputIndex]));
            impl_->cudaOccupied=false;
        }
#ifdef TRT7
        impl_->context->destroy();
        impl_->engine->destroy();
        impl_->runtime->destroy();
#else
        delete impl_->context;
        delete impl_->engine;
        delete impl_->runtime;
        impl_->context = nullptr;
        impl_->engine = nullptr;
        impl_->runtime = nullptr;
#endif
        impl_->init=false;
        delete impl_;
        impl_ = nullptr;
    }
}


#endif
#define YOLO_TENSORRT_CPP