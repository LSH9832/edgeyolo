#ifndef YOLO_HORIZON_CPP
#define YOLO_HORIZON_CPP


#include "./dnn/hb_dnn.h"


#include "../../yolo.hpp"
#include "../../str.h"
#include "../../datetime.h"
#include "../common.hpp"
#include <string.h>




/* ----------------------- implimentation ----------------------- */

void platform(char* p)
{
    std::string p_ = "Horizon";
    for (int i=0;i<p_.size();i++)
    {
        p[i] = p_[i];
    }
}


struct YOLO::Impl
{
    hbPackedDNNHandle_t packed_dnn_handle;
    hbDNNHandle_t dnn_handle;
    hbDNNInferCtrlParam infer_ctrl_param;
    std::vector<hbDNNTensor> netInputs, netOutputs;

    int num_inputs=0, num_outputs=0;
    std::vector<int> outFlags;
    bool nchw=true;

    std::vector<int> gridsX, gridsY;

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

    // bool RC_Mode=false;
    int32_t deviceID=0;

    // std::vector<int> rerank = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

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
    if (key == "input")
    {
        pystring type = pystring(value).lower();
        if (type == "nchw")
        {
            impl_->nchw = true;
        }
        else if (type == "nhwc")
        {
            impl_->nchw = false;
        }
        else
        {
            std::cout << "unknown input type '" << value << "'" << std::endl;
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
    else if (key == "show_info")
    {
        impl_->showInfo = pystring(value);
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

    int model_count = 0;
    const char* modelFileName = modelFile_.c_str();
    const char **model_name_list;
    if (hbDNNInitializeFromFiles(&impl_->packed_dnn_handle, &modelFileName, 1) != 0) {
        std::cerr << "[E] hbDNNInitializeFromFiles failed" << std::endl;
        return false;
    }
    if (hbDNNGetModelNameList(&model_name_list, &model_count, impl_->packed_dnn_handle) != 0) {
        std::cerr << "[E] hbDNNGetModelNameList failed" << std::endl;
        return false;
    }
    if (hbDNNGetModelHandle(&impl_->dnn_handle, impl_->packed_dnn_handle, model_name_list[0]) != 0) {
        std::cerr << "[E] hbDNNGetModelHandle failed" << std::endl;
        return false;
    }

    if (hbDNNGetInputCount(&impl_->num_inputs, impl_->dnn_handle) != 0) {
        std::cerr << "[E] hbDNNGetInputCount failed" << std::endl;
        return false;
    }
    if (hbDNNGetOutputCount(&impl_->num_outputs, impl_->dnn_handle) != 0) {
        std::cerr << "[E] hbDNNGetOutputCount failed" << std::endl;
        return false;
    }

    if (impl_->num_inputs != 1)
    {
        std::cerr << "[E] only support 1 input model now, but got " << impl_->num_inputs << std::endl;
        return false;
    }
    const char *inputNamecstr;
    hbDNNGetInputName(&inputNamecstr, impl_->dnn_handle, 0);
    std::string hbDNNInputName = inputNamecstr;
    if (inputName_ != hbDNNInputName)
    {
        std::cerr << "[E] model input name is '" << hbDNNInputName << "' but set '" << inputName_ << "'" << std::endl;
        return false;
    }

    if (impl_->num_outputs != outputNames_.size())
    {
        std::cerr << "[W] this model has " << impl_->num_outputs << " but set " << outputNames_.size() << std::endl;
        if (impl_->num_outputs < outputNames_.size()) return false;
    }
    impl_->outFlags.resize(outputNames_.size());
    for (int i=0;i<impl_->outFlags.size();i++) impl_->outFlags[i] = -1;
    for (int i=0;i<impl_->num_outputs;i++)
    {
        const char* thisOutputNamecstr;
        hbDNNGetOutputName(&thisOutputNamecstr, impl_->dnn_handle, i);
        std::string thishbDNNOutputName = thisOutputNamecstr;
        for(int j=0;j<outputNames_.size();j++)
        {
            if (thishbDNNOutputName == outputNames_[j])
            {
                impl_->outFlags[j] = i;
            }
        }
    }
    for(int j=0;j<outputNames_.size();j++)
    {
        if (impl_->outFlags[j] < 0)
        {
            std::cerr << "can not find output name: '" << outputNames_[j] << "'" << std::endl;
            return false;
        }
    }
    
    if (impl_->showInfo)
    {
        std::cout << "num inputs: " << impl_->num_inputs << std::endl;
        std::cout << "num outputs: " << impl_->num_outputs << std::endl;
    }
    
                    
    impl_->netInputs.resize(impl_->num_inputs);
    impl_->netOutputs.resize(outputNames_.size());

    for (int i=0;i<impl_->num_inputs;i++) {
        if (hbDNNGetInputTensorProperties(&impl_->netInputs[i].properties, impl_->dnn_handle, i) != 0) {
            std::cerr << "[E] hbDNNGetInputTensorProperties failed" << std::endl;
            return false;
        }

        if (hbSysAllocCachedMem(&impl_->netInputs[i].sysMem[0], impl_->netInputs[i].properties.alignedByteSize) != 0) 
        {
            std::cerr << "[E] hbSysAllocCachedMem for inputs failed" << std::endl;
            return false;
        }

        bool nchw = true;
        
        switch (impl_->netInputs[i].properties.tensorLayout) 
        {
            case HB_DNN_IMG_TYPE_BGR:
                std::cout << "input layout: BGR" << std::endl;
                nchw = false;
                break;
            case HB_DNN_LAYOUT_NHWC:
                std::cout << "input layout: NHWC" << std::endl;
                nchw = false;
                break;
            case HB_DNN_LAYOUT_NCHW:
                std::cout << "input layout: NCHW" << std::endl;
                break;
            default:
                std::cout << "no rule for tensorLayout, use default nchw." << std::endl;
                break;
        }
        if (nchw != impl_->nchw)
        {
            std::cerr << "[W] input type should be " << (nchw?"NCHW":"NHWC") << " but set "
                      << (impl_->nchw?"NCHW":"NHWC") << ", please make sure you add transfrom layer." << std::endl;
            // return false;
        }
        
        int numDimInput = impl_->netInputs[i].properties.alignedShape.numDimensions;
        bool bgr = (numDimInput == 3);
        if (!bgr && numDimInput != 4)
        {
            std::cerr << "[E] num of input Dimensions should be 4 but got " 
                      << impl_->netInputs[i].properties.alignedShape.numDimensions << "."
                      << std::endl;
            return false;
        }

        if (bgr)
        {
            impl_->batch = 1;
            int imgh_ = impl_->netInputs[i].properties.alignedShape.dimensionSize[0];
            int imgw_ = impl_->netInputs[i].properties.alignedShape.dimensionSize[1];
            impl_->imgChannel = impl_->netInputs[i].properties.alignedShape.dimensionSize[2];
            if (imgH_ != imgh_)
            {
                std::cerr << "[E] model input image height is " << imgh_
                            << " but set " << imgH_ << std::endl;
                return false; 
            }
            if (imgW_ != imgw_)
            {
                std::cerr << "[E] model input image width is " << imgw_
                            << " but set " << imgW_ << std::endl;
                return false; 
            }
        }
        else
        {
            for (int c=0;c<impl_->netInputs[i].properties.alignedShape.numDimensions;c++) 
            {
                int dim_len = impl_->netInputs[i].properties.alignedShape.dimensionSize[c];
                if (c == 0)
                {
                    impl_->batch = dim_len;
                    if (dim_len != 1)
                    {
                        std::cerr << "[E] only support batch 1 model currently." << std::endl;
                    }
                }
                if (c == (nchw?1:3))
                {
                    impl_->imgChannel = dim_len;
                    if (dim_len != 3)
                    {
                        std::cout << "[W] please notice image channel is " << dim_len << std::endl;
                    }
                }
                if (c==(nchw?2:1)) 
                {
                    if (imgH_ != dim_len)
                    {
                        std::cerr << "[E] model input image height is " << dim_len
                                    << " but set " << imgH_ << std::endl;
                        return false; 
                    }
                }
                else if (c==(nchw?3:2)) 
                {
                    if (imgW_ != dim_len)
                    {
                        std::cerr << "[E] model input image width is " << dim_len
                                    << " but set " << imgW_ << std::endl;
                        return false; 
                    }
                }
            }
        }
        
        if (impl_->showInfo) std::cout << "[I] input data type: " << impl_->netInputs[i].properties.tensorType << std::endl;
    }

    impl_->inputSize = impl_->imgChannel * imgH_ * imgW_;

    // hbDNNTensor *output = impl_->netOutputs.data();
    for (int i=0;i<outputNames_.size();i++) 
    {
        int actualIdx = impl_->outFlags[i];
        if (hbDNNGetOutputTensorProperties(&impl_->netOutputs[i].properties, impl_->dnn_handle, actualIdx) != 0) {
            std::cerr << "[E] hbDNNGetOutputTensorProperties failed" << std::endl;
            return false;
        }

        if (hbSysAllocCachedMem(&impl_->netOutputs[i].sysMem[0], impl_->netOutputs[i].properties.alignedByteSize) != 0) {
            std::cerr << "[E] hbSysAllocCachedMem for outputs failed" << std::endl;
            return false;
        }

        if (impl_->showInfo) 
        {
            std::cout << "output '" << outputNames_[i] << "' data type: " 
                      << impl_->netOutputs[i].properties.tensorType << std::endl;
        }
            
    }

    impl_->needDecoder = outputNames_.size() > 1;
    if (impl_->needDecoder) {
        if (outputNames_.size() == strides_.size() * 3)   // 9 in general
        {
            impl_->splitHead = true;
            impl_->lengthArray = impl_->netOutputs[strides_.size() * 2].properties.alignedShape.dimensionSize[1];
            impl_->numClasses = impl_->lengthArray;
        }
        else if (outputNames_.size() == strides_.size()) // 3 in general
        {
            impl_->splitHead = false;
            impl_->lengthArray = impl_->netOutputs[0].properties.alignedShape.dimensionSize[1];
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
        impl_->numArrays = impl_->netOutputs[0].properties.alignedShape.dimensionSize[1];
        impl_->lengthArray = impl_->netOutputs[0].properties.alignedShape.dimensionSize[2];
        impl_->numClasses = impl_->lengthArray - (impl_->obj_conf_enabled?5:4);
    }


    // INFO << 3 << std::endl;
    // ------------------------------------------------------------------------------------
    if (impl_->needDecoder) {
        impl_->numArrays = 0;
        for(int i=0; i<outputNames_.size(); i++) 
        {
            if (impl_->splitHead && i < strides_.size() * 2) continue;
            int gx = impl_->netOutputs[i].properties.alignedShape.dimensionSize[3];
            int gy = impl_->netOutputs[i].properties.alignedShape.dimensionSize[2];
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
    return impl_->netInputs[0].sysMem[0].virAddr;
}


void YOLO::inference(void* data, void* preds, float scale)
{
    
    if (data != nullptr)
    {
        if (data != getInputData())
        {
            memcpy(impl_->netInputs[0].sysMem[0].virAddr, data, impl_->inputSize);
        }
    }

    double t0 = 0, tInfer = 0.;
    if (impl_->showInfo) t0 = pytime::time();


    // -------------- inference --------------
    for (int i=0; i<impl_->num_inputs; i++) {
        hbSysFlushMem(&impl_->netInputs[i].sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
    }

    hbDNNTensor *outputs = impl_->netOutputs.data();
    hbDNNTaskHandle_t task_handle = nullptr;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&impl_->infer_ctrl_param);
    int ret = hbDNNInfer(
        &task_handle, &outputs, impl_->netInputs.data(), 
        impl_->dnn_handle, &impl_->infer_ctrl_param
    );
    if (ret != 0) 
    {
        std::cerr << "[E] hbDNNInfer failed with code " << ret << std::endl;
        return;
    }
                     
    // wait task done
    if (hbDNNWaitTaskDone(task_handle, 0) == 0) {}
    else {
        std::cerr << "[E] hbDNNWaitTaskDone failed" << std::endl;
        return;
    }

    // make sure CPU read data from DDR before using output tensor data
    for (int i=0; i<outputNames_.size(); i++) 
    {
        hbSysFlushMem(&impl_->netOutputs[i].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    }

    if (hbDNNReleaseTask(task_handle) != 0)
    {
        std::cerr << "hbDNNReleaseTask Failed" << std::endl;
        return;
    }

    if (impl_->showInfo) 
    {
        tInfer = pytime::time() - t0;
        t0 = pytime::time();
    }

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
                    impl_->netOutputs[i + numStrides * 0].sysMem[0].virAddr,
                    impl_->netOutputs[i + numStrides * 1].sysMem[0].virAddr,
                    impl_->netOutputs[i + numStrides * 2].sysMem[0].virAddr,
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
                    impl_->netOutputs[i].sysMem[0].virAddr,
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
            (float*)impl_->netOutputs[0].sysMem[0].virAddr,
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
    
    if (impl_->showInfo)
    {
        std::cout << std::fixed << std::setprecision(3) 
                  << "infer: " << tInfer * 1000 << "ms, post-process: " 
                  << (pytime::time() - t0) * 1000 << "ms." << std::endl;
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