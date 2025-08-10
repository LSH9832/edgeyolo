#ifndef OPENVINO_YOLO_CPP
#define OPENVINO_YOLO_CPP

#include <openvino/openvino.hpp>

#include "../common.h"
#include "../../yolo.hpp"
#include "pylike/str.h"
#include "pylike/datetime.h"


void platform(char* p)
{
    std::string p_ = "OpenVINO";
    for (int i=0;i<p_.size();i++)
    {
        p[i] = p_[i];
    }
}


struct YOLO::Impl
{
    ov::Core core;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel cmodel;
    ov::InferRequest infer_request;

    bool enableASYNC = false;
    bool fp16 = true;

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
    std::string deviceName = "CPU";

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
    impl_->deviceName = impl_->core.get_available_devices()[impl_->deviceID];
}

void YOLO::set(std::string key, std::string value)
{
    // std::cout << key << "," << value << std::endl;
    if (key == "anchors")
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
    else if (key == "device")
    {
        impl_->deviceName = value;
    }
    else if (key == "async")
    {
        impl_->enableASYNC = pystring(value);
    }
    else if (key == "dtype")
    {
        impl_->fp16 = value == "fp16";
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
    // load model
    pystring mf = pystring(modelFile_).lower();

    std::cout << "use device: " << impl_->deviceName << std::endl;
    
    if (mf.endswith(".onnx") || mf.endswith(".tf") || 
        mf.endswith(".tflite") || mf.endswith(".ir") || mf.endswith(".pdpd"))
    {
        auto dataType = impl_->fp16?ov::element::f16:ov::element::f32;
        impl_->model = impl_->core.read_model(modelFileName);
        impl_->cmodel = impl_->core.compile_model(
            impl_->model, impl_->deviceName, 
            ov::hint::inference_precision(dataType)
        );
    }
    else if (mf.endswith(".xml"))
    {
        impl_->cmodel = impl_->core.compile_model(modelFileName, impl_->deviceName);
        // impl_->model = impl_->cmodel.get_model();
    }
    else
    {
        std::cerr << "[E] unsupported model format: " << modelFile_ << std::endl;
        return false;
    }

    impl_->num_inputs = impl_->cmodel.inputs().size();
    impl_->num_outputs = impl_->cmodel.outputs().size();
    if (impl_->num_inputs != 1)
    {
        std::cerr << "[E] only support 1 input model now, but got " << impl_->num_inputs << std::endl;
        return false;
    }

    std::string ovInputName = impl_->cmodel.input(0).get_any_name();
    if (impl_->showInfo) std::cout << "model input name = " << ovInputName << std::endl;
    if (inputName_ != ovInputName)
    {
        std::cerr << "[E] model input name is '" << ovInputName << "' but set '" << inputName_ << "'" << std::endl;
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
        std::string thisOvOutputName = impl_->cmodel.output(i).get_any_name();
        for(int j=0;j<outputNames_.size();j++)
        {
            if (thisOvOutputName == outputNames_[j])
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
    
                    
    // impl_->netInputs.resize(impl_->num_inputs);
    // impl_->netOutputs.resize(outputNames_.size());

    for (int i=0;i<impl_->num_inputs;i++) 
    {
        int numDimInput = impl_->cmodel.input(i).get_partial_shape().size();
        if (numDimInput != 4)
        {
            std::cerr << "[E] num of input Dimensions should be 4 but got " 
                      << numDimInput << "."
                      << std::endl;
            return false;
        }

        bool nchw = true;
        for (int c=0;c<numDimInput;c++) 
        {
            int dim_len = impl_->cmodel.input(i).get_partial_shape()[c].get_length(); // impl_->netInputs[i].properties.alignedShape.dimensionSize[c];
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

    impl_->inputSize = impl_->imgChannel * imgH_ * imgW_;


    impl_->needDecoder = outputNames_.size() > 1;
    if (impl_->needDecoder) {
        if (outputNames_.size() == strides_.size() * 3)   // 9 in general
        {
            impl_->splitHead = true;
            // impl_->netOutputs[strides_.size() * 2].properties.alignedShape.dimensionSize[1];
            impl_->lengthArray = impl_->cmodel.output(strides_.size() * 2).get_partial_shape()[1].get_length();
            impl_->numClasses = impl_->lengthArray;
        }
        else if (outputNames_.size() == strides_.size()) // 3 in general
        {
            impl_->splitHead = false;
            // impl_->netOutputs[0].properties.alignedShape.dimensionSize[1];
            impl_->lengthArray = impl_->cmodel.output(0).get_partial_shape()[1].get_length();
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
        // impl_->netOutputs[0].properties.alignedShape.dimensionSize[1];
        impl_->numArrays = impl_->cmodel.output(0).get_partial_shape()[1].get_length();
        // impl_->netOutputs[0].properties.alignedShape.dimensionSize[2];
        impl_->lengthArray = impl_->cmodel.output(0).get_partial_shape()[2].get_length();
        impl_->numClasses = impl_->lengthArray - (impl_->obj_conf_enabled?5:4);
    }


    // INFO << 3 << std::endl;
    // ------------------------------------------------------------------------------------
    if (impl_->needDecoder) {
        impl_->numArrays = 0;
        for(int i=0; i<outputNames_.size(); i++) 
        {
            if (impl_->splitHead && i < strides_.size() * 2) continue;
            // impl_->netOutputs[i].properties.alignedShape.dimensionSize[3];
            int gx = impl_->cmodel.output(i).get_partial_shape()[3].get_length();
            // impl_->netOutputs[i].properties.alignedShape.dimensionSize[2];
            int gy = impl_->cmodel.output(i).get_partial_shape()[2].get_length();
            impl_->gridsX.push_back(gx);
            impl_->gridsY.push_back(gy);
            impl_->numArrays += impl_->numAnchors * gx * gy;
        }
    }

    impl_->outputSize = impl_->batch * impl_->numArrays * impl_->lengthArray;

    impl_->infer_request = impl_->cmodel.create_infer_request();

    if (impl_->infer_request.get_input_tensor(0).get_size() != impl_->inputSize)
    {
        std::cerr << "[E] can not init net: " << "model input size is "
            << impl_->infer_request.get_input_tensor(0).get_size()
            << " but set " << impl_->inputSize << std::endl;
        return false;
    }

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

void* YOLO::getInputData() 
{
    if (!inputDataReachable()) return nullptr;
    return impl_->infer_request.get_input_tensor(0).data();    
}


void YOLO::inference(void* data, void* preds, float scale)
{
    
    if (data != nullptr)
    {
        if (data != getInputData())
        {
            memcpy(impl_->infer_request.get_input_tensor(0).data(), data, sizeof(float) * impl_->inputSize);
        }
    }

    double t0 = 0, tInfer = 0.;
    if (impl_->showInfo) t0 = pytime::time();


    // -------------- inference --------------
    if (impl_->enableASYNC)
    {
        impl_->infer_request.start_async();
        impl_->infer_request.wait();
    }
    else
    {
        impl_->infer_request.infer();
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
                    impl_->infer_request.get_output_tensor(i + numStrides * 0).data(),
                    impl_->infer_request.get_output_tensor(i + numStrides * 1).data(),
                    impl_->infer_request.get_output_tensor(i + numStrides * 2).data(),
                    // impl_->netOutputs[i + numStrides * 0].sysMem[0].virAddr,
                    // impl_->netOutputs[i + numStrides * 1].sysMem[0].virAddr,
                    // impl_->netOutputs[i + numStrides * 2].sysMem[0].virAddr,
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
                    // impl_->netOutputs[i].sysMem[0].virAddr,
                    impl_->infer_request.get_output_tensor(i).data(),
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
            // (float*)impl_->netOutputs[0].sysMem[0].virAddr,
            (float*)impl_->infer_request.get_output_tensor(0).data(),
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


#endif // OPENVINO_YOLO_CPP