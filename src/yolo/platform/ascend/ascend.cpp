#ifndef YOLO_ASCEND_CPP
#define YOLO_ASCEND_CPP

#include "./davinci_net.h"

#include "../../yolo.hpp"
#include "../../str.h"
#include "../../datetime.h"
#include "../common.hpp"
#include <string.h>

/* ----------------------- implimentation ----------------------- */

void platform(char* p)
{
    std::string p_ = "Ascend";
    for (int i=0;i<p_.size();i++)
    {
        p[i] = p_[i];
    }
}


struct YOLO::Impl
{
    DavinciNet davinciNet;

    std::shared_ptr<Blob<float>> data;
    std::shared_ptr<Blob<float>> preds;
    std::vector<std::shared_ptr<Blob<float>>> sp_preds;


    std::vector<int8_t> conf_thres_int8s;
    std::vector<int> all_num_dets;
    std::vector<int> gridsX, gridsY;

    std::vector<int32_t> output_zps;
    std::vector<float> output_scales;
    uint8_t outputFloat = 0;

    bool regExp=true;
    bool regMul=false;

    // common
    bool first=true;
    bool init=false;
    bool obj_conf_enabled=true;
    bool deviceOccupied=false;
    long long outputSize = 0;
    int batch = 1;
    int numArrays=0;
    int lengthArray=0;
    int numClasses=0;

    int imgChannel = 3;
    long inputSize = 0;

    float confThres=0.25, nmsThres=0.45;

    bool anchorBased = false;
    int numAnchors = 1;      // anchor free = 1
    bool needDecoder=true;   // num of outputs > 1
    bool splitHead=true;     // num of outputs == num strides * 3
    bool showInfo=false;

    std::vector<std::vector<std::vector<float>>> anchors;;

    bool RC_Mode=false;
    int32_t deviceID=0;

    std::vector<int> rerank = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

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
    outputNames_ = outputNames;
    imgW_ = imgW;
    imgH_ = imgH;
    strides_ = strides;
    if (impl_ == nullptr)
    {
        impl_ = new Impl();
    }
    impl_->deviceID = device;
}


void YOLO::set(std::string key, std::string value)
{
    // std::cout << key << "," << value << std::endl;
    if (key == "rc_mode")
    {
        impl_->RC_Mode = (bool)pystring(value);
    }
    else if (key == "rerank")
    {
        impl_->rerank.clear();
        for (pystring rk: pystring(value).split(","))
        {
            impl_->rerank.push_back(rk);
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

    aclError ret = aclInit(nullptr);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "[E] acl init failed!" << std::endl;
        exit(-1);
    }

    ret = aclrtSetDevice(impl_->deviceID);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "[E] acl set device failed!" << std::endl;
        exit(-1);
    }

    ret = impl_->davinciNet.Init(modelFile_, impl_->deviceID);
    if (ret != 0) {
        std::cerr << "[E] Davinci Net init failed!" << std::endl;
        exit(-1);
    }

    inputName_ = impl_->davinciNet.input_names[0];
    impl_->batch = impl_->davinciNet.inputs_dims[0][0];
    impl_->imgChannel = impl_->davinciNet.inputs_dims[0][1];
    impl_->inputSize = impl_->imgChannel * imgH_ * imgW_;
    if (imgH_ != impl_->davinciNet.inputs_dims[0][2])
    {
        std::cerr << "[W] input height should be " << impl_->davinciNet.inputs_dims[0][2]
                  << " but set " << imgH_ << ", change to correct." << std::endl;
        imgH_ = impl_->davinciNet.inputs_dims[0][2];
    }
    if (imgW_ != impl_->davinciNet.inputs_dims[0][3])
    {
        std::cerr << "[W] input width should be " << impl_->davinciNet.inputs_dims[0][3]
                  << " but set " << imgW_ << ", change to correct." << std::endl;
        imgW_ = impl_->davinciNet.inputs_dims[0][3];
    }

    // INFO << 2 << ENDL;

    impl_->needDecoder = impl_->davinciNet.output_names.size() > 1;
    if (impl_->needDecoder) {
        outputNames_.resize(impl_->davinciNet.output_names.size());
        for (int i=0;i<impl_->davinciNet.output_names.size();i++) {
            outputNames_[i] = impl_->davinciNet.output_names[impl_->rerank[i]];
        }

        if (impl_->davinciNet.output_names.size()==strides_.size() * 3) {
            impl_->splitHead = true;
            impl_->lengthArray = impl_->davinciNet.outputs_dims[strides_.size() * 2][1];
            impl_->numClasses = impl_->lengthArray;
        }
        else if (impl_->davinciNet.output_names.size()==strides_.size()) {
            impl_->splitHead = false;
            impl_->lengthArray = impl_->davinciNet.outputs_dims[0][1];
            if (impl_->anchors.size()) {
                impl_->lengthArray /= impl_->anchors[0].size();
            }
            impl_->numClasses = impl_->lengthArray - (impl_->obj_conf_enabled?5:4);
        }
        else {
            std::cerr << "[E] num outs is " << impl_->davinciNet.output_names.size() << ", which is not supported." << std::endl;
            exit(-1);
        }

    }
    else {
        outputNames_ = impl_->davinciNet.output_names;
        impl_->numArrays = impl_->davinciNet.outputs_dims[0][1];
        impl_->lengthArray = impl_->davinciNet.outputs_dims[0][2];
        impl_->numClasses = impl_->lengthArray - (impl_->obj_conf_enabled?5:4);
    }


    // INFO << 3 << ENDL;

    impl_->data = std::static_pointer_cast<Blob<float>>(impl_->davinciNet.GetBlob(inputName_));
    if (impl_->needDecoder) {
        impl_->sp_preds.resize(outputNames_.size());
        if (impl_->splitHead) {
            impl_->numArrays = 0;
            for(int i=0; i<outputNames_.size(); i++) {
                impl_->sp_preds[i] = std::static_pointer_cast<Blob<float>>(impl_->davinciNet.GetBlob(outputNames_[i]));
                if (i >= strides_.size() * 2) {
                    int gx = impl_->davinciNet.outputs_dims[impl_->rerank[i]][3];
                    int gy = impl_->davinciNet.outputs_dims[impl_->rerank[i]][2];
                    impl_->gridsX.push_back(gx);
                    impl_->gridsY.push_back(gy);
                    impl_->numArrays += impl_->numAnchors * gx * gy;
                }
            }
        }
        else {
            impl_->numArrays = 0;
            for(int i=0; i<outputNames_.size(); i++) {
                int gx = impl_->davinciNet.outputs_dims[impl_->rerank[i]][3];
                int gy = impl_->davinciNet.outputs_dims[impl_->rerank[i]][2];

                impl_->sp_preds[i] = std::static_pointer_cast<Blob<float>>(impl_->davinciNet.GetBlob(outputNames_[i]));
                impl_->gridsX.push_back(gx);
                impl_->gridsY.push_back(gy);
                impl_->numArrays += impl_->numAnchors * gx * gy;
            }
        }
    }
    else {
        impl_->sp_preds.resize(1);
        impl_->sp_preds[0] = std::static_pointer_cast<Blob<float>>(impl_->davinciNet.GetBlob(outputNames_[0]));
    }

    // impl_->numArrays = impl_->outputTensor->shape().at(1);
    // impl_->numClasses = impl_->outputTensor->shape().at(2) - (impl_->obj_conf_enabled?5:4);


    // impl_->batch = impl_->inputTensor->shape().at(0);
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

void* YOLO::getInputData() {
    if (!inputDataReachable()) return nullptr;
    if (impl_->RC_Mode) return impl_->data->Data();
    else return impl_->data->HostData();
}


void YOLO::inference(void* data, void* preds, float scale)
{
    if (data != nullptr)
    {
        float* data2process = (float*)data;
        if (data2process != getInputData())
        {
            memcpy(getInputData(), data2process, sizeof(float) * impl_->inputSize);
        }
    }

    if (!impl_->RC_Mode) {
        impl_->data->dataHost2Device();
    }
    int ret = impl_->davinciNet.Inference();
    if (ret != 0) {
        std::cerr << "[E] Davinci Net inference failed!" << std::endl;
        exit(-1);
    }

    std::vector<float*> net_outs;
    if (impl_->RC_Mode) {
        for(int i=0;i<impl_->sp_preds.size();i++) 
            net_outs.push_back(impl_->sp_preds[i]->Data());
    }
    else {
        for(int i=0;i<impl_->sp_preds.size();i++) 
            net_outs.push_back(impl_->sp_preds[i]->HostData());
    }

    // t0 = pytime::time();
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
                    (impl_->RC_Mode?impl_->sp_preds[i + numStrides * 0]->Data():impl_->sp_preds[i + numStrides * 0]->HostData()),
                    (impl_->RC_Mode?impl_->sp_preds[i + numStrides * 1]->Data():impl_->sp_preds[i + numStrides * 1]->HostData()),
                    (impl_->RC_Mode?impl_->sp_preds[i + numStrides * 2]->Data():impl_->sp_preds[i + numStrides * 2]->HostData()),
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
                    (impl_->RC_Mode?impl_->sp_preds[i]->Data():impl_->sp_preds[i]->HostData()), 
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
            (float*)(impl_->RC_Mode?impl_->sp_preds[0]->Data():impl_->sp_preds[0]->HostData()), 
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
    delete impl_;
    impl_ = nullptr;
}


#endif