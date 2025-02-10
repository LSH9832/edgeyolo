#ifndef YOLO_RKNN_CPP
#define YOLO_RKNN_CPP

#include <iostream>
#include <sstream>
#include <vector>
#include <string.h>
#include <math.h>

#include "../../yolo.hpp"
#include "../../str.h"
#include "../../datetime.h"
#include "../common.hpp"
#include "./rknn_api.h"


static void dump_tensor_attr(rknn_tensor_attr *attr) 
{
    std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
    for (int i = 1; i < attr->n_dims; ++i) shape_str += ", " + std::to_string(attr->dims[i]);

    printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, w_stride = %d, size_with_stride=%d, fmt=%s, "
            "type=%s, qnt_type=%s, "
            "zp=%d, scale=%f\n",
            attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->n_elems, attr->size, attr->w_stride,
            attr->size_with_stride, get_format_string(attr->fmt), get_type_string(attr->type),
            get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}


/* ----------------------- implimentation ----------------------- */

void platform(char* p)
{
    std::string p_ = "RKNN";
    for (int i=0;i<p_.size();i++)
    {
        p[i] = p_[i];
    }
}


struct YOLO::Impl
{
    // rknn
    rknn_context ctx;
    rknn_core_mask mode=RKNN_NPU_CORE_0_1_2;
    rknn_tensor_format fmt = RKNN_TENSOR_NHWC;
    rknn_tensor_type dtype = RKNN_TENSOR_UINT8;
    std::vector<rknn_input> inputs;
    std::vector<rknn_output> outputs;

    std::vector<int8_t> conf_thres_int8s;
    std::vector<int> all_num_dets;
    std::vector<int> all_x_grids;
    std::vector<int> all_y_grids;

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
    int numClasses=0;

    float confThres=0.25, nmsThres=0.45;

    bool anchorBased = false;
    int numAnchors = 1;      // anchor free = 1
    bool needDecoder=true;   // num of outputs > 1
    bool splitHead=true;     // num of outputs == num strides * 3
    bool showInfo=false;

    std::vector<std::vector<std::vector<float>>> anchors;;

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
}

void YOLO::set(std::string key, std::string value)
{
    // std::cout << key << "," << value << std::endl;
    if (key == "core")
    {
        pystring type = pystring(value).lower();
        if (type == "core0")
        {
            impl_->mode = RKNN_NPU_CORE_0;
        }
        else if (type == "core1")
        {
            impl_->mode = RKNN_NPU_CORE_1;
        }
        else if (type == "core2")
        {
            impl_->mode = RKNN_NPU_CORE_2;
        }
        else if (type == "core01")
        {
            impl_->mode = RKNN_NPU_CORE_0_1;
        }
        else if (type == "core012")
        {
            impl_->mode = RKNN_NPU_CORE_0_1_2;
        }
        else if (type == "all")
        {
            impl_->mode = RKNN_NPU_CORE_ALL;
        }
        else if (type == "auto")
        {
            impl_->mode = RKNN_NPU_CORE_AUTO;
        }
        else
        {
            std::cout << "unknown core type '" << value << "'" << std::endl;
        }
    }
    else if (key == "input")
    {
        pystring type = pystring(value).lower();
        if (type == "nchw")
        {
            impl_->fmt = RKNN_TENSOR_NCHW;
        }
        else if (type == "nhwc")
        {
            impl_->fmt = RKNN_TENSOR_NHWC;
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
        impl_->showInfo = (bool)pystring(value);
    }
    
    else
    {
        std::cout << "unused setting -> " << key << ": " << value << std::endl;
    }
}

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz) {
    unsigned char *data;
    int ret;
    data = NULL;
    if (NULL == fp) return NULL;

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0) {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL) {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
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

    // load data from file
    FILE *fp;
    unsigned char *data;
    fp = fopen(modelFile_.c_str(), "rb");
    if (NULL == fp) 
    {
        std::cerr << "failed to open " << modelFile_ << std::endl;
        return false;
    }
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);
    data = load_data(fp, 0, size);
    
    fclose(fp);

    // load data to model
    int ret = rknn_init(&impl_->ctx, data, size, 0, NULL);
    if (ret < 0)
    {
        std::cerr << "rknn init error code: " << ret << std::endl;
        return false;
    }
    ret = rknn_set_core_mask(impl_->ctx, impl_->mode);
    if (ret < 0) {
        std::cerr << "rknn_init core error code: " << ret << std::endl;
        return false;
    }

    rknn_input_output_num io_num;
    ret = rknn_query(impl_->ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        return false;
    }
    if (impl_->showInfo) printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    if (io_num.n_input != 1)
    {
        std::cerr << "only support 1 input but got " << io_num.n_input << std::endl;
        return false;
    }
    // deal input
    int channel = 3;
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(impl_->ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            std::cerr << "rknn query input attr error code: " << ret << std::endl;
            return false;
        }
        if (impl_->showInfo) dump_tensor_attr(&(input_attrs[i]));

        if (input_attrs[i].n_dims != 4)
        {
            std::cerr << "only support input with 4 dims but got " << input_attrs[i].n_dims << std::endl;
            return false;
        }
        
        impl_->batch = input_attrs[i].dims[0];
        if (input_attrs[i].fmt == RKNN_TENSOR_NCHW)
        {
            channel = input_attrs[i].dims[1];
            if (imgH_ != input_attrs[i].dims[2])
            {
                std::cout << "[W] input height should be " << input_attrs[i].dims[2] 
                          << " rather than " << imgH_ << ", change." << std::endl;
                imgH_ = input_attrs[i].dims[2];
            }
            if (imgW_ != input_attrs[i].dims[3])
            {
                std::cout << "[W] input width should be " << input_attrs[i].dims[3] 
                          << " rather than " << imgW_ << ", change." << std::endl;
                imgW_ = input_attrs[i].dims[3];
            }

        }
        else if (input_attrs[i].fmt == RKNN_TENSOR_NHWC)
        {
            /* code */
            imgH_ = input_attrs[i].dims[1];
            imgW_ = input_attrs[i].dims[2];
            if (imgH_ != input_attrs[i].dims[1])
            {
                std::cout << "[W] input height should be " << input_attrs[i].dims[1] 
                          << " rather than " << imgH_ << ", change." << std::endl;
                imgH_ = input_attrs[i].dims[1];
            }
            if (imgW_ != input_attrs[i].dims[2])
            {
                std::cout << "[W] input width should be " << input_attrs[i].dims[2] 
                          << " rather than " << imgW_ << ", change." << std::endl;
                imgW_ = input_attrs[i].dims[2];
            }
            channel = input_attrs[i].dims[3];
        }
        else
        {
            std::cerr << "only support input format NCHW and NHWC" << std::endl;
            return false;
        }
        if (channel != 3)
        {
            std::cout << "[W] channel should be 3 but got " << channel 
                      << ", which may cause errors." << std::endl;
        }
        
    }

    // deal output
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));

    impl_->obj_conf_enabled = true;
    impl_->numArrays = 0;
    if (io_num.n_output == 1)
    {
        impl_->needDecoder = false;
        impl_->splitHead = false;
        // todo
        output_attrs[0].index = 0;
        ret = rknn_query(impl_->ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[0]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(output_attrs[0]));
        impl_->output_zps.push_back(output_attrs[0].zp);
        impl_->output_scales.push_back(output_attrs[0].scale);
        int n_dims = output_attrs[0].n_dims;
        if (n_dims == 3) // batch, numArrays, lengthArray
        {
            impl_->numClasses = output_attrs[0].dims[2] / impl_->numAnchors - (impl_->obj_conf_enabled?5:4);
            impl_->numArrays = output_attrs[0].dims[1];
            impl_->outputSize = output_attrs[0].dims[0] * output_attrs[0].dims[1] * output_attrs[0].dims[2];
        }
        else if (n_dims == 4) // batch, anchor, numArrays, lengthArray
        {
            /* code */
            if (output_attrs[0].dims[1] != impl_->numAnchors)
            {
                std::cerr << "num anchors not match, should be " << output_attrs[0].dims[1]
                          << "but got " << impl_->numAnchors << std::endl;
                return false;
            }
            impl_->numClasses = output_attrs[0].dims[3] - (impl_->obj_conf_enabled?5:4);
            impl_->numArrays = output_attrs[0].dims[1] * output_attrs[0].dims[2];
            impl_->outputSize = output_attrs[0].dims[0] * output_attrs[0].dims[1] * 
                                output_attrs[0].dims[2] * output_attrs[0].dims[3];
        }
        else
        {
            std::cerr << "wrong number of output dims, which should be 3 or 4 but got " << n_dims << std::endl;
            return false;
        }
    }
    else if (io_num.n_output == strides_.size())
    {
        impl_->needDecoder = true;
        impl_->splitHead = false;

        // todo
        for (int i = 0; i < io_num.n_output; i++) 
        {
            output_attrs[i].index = i;
            ret = rknn_query(impl_->ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
            dump_tensor_attr(&(output_attrs[i]));
            impl_->output_zps.push_back(output_attrs[i].zp);
            impl_->output_scales.push_back(output_attrs[i].scale);

            int n_dims = output_attrs[i].n_dims;  
            if(!i)
            {
                if (n_dims == 4) // batch, lengthArray, h, w
                {
                    impl_->numClasses = output_attrs[i].dims[n_dims-3] / impl_->numAnchors - (impl_->obj_conf_enabled?5:4);
                }
                else if (n_dims == 5)   // batch, anchor, lengthArray, h, w
                {
                    /* code */
                    if (impl_->numAnchors != output_attrs[i].dims[1])
                    {
                        std::cerr << "num anchors not match, should be " << output_attrs[i].dims[1]
                                  << "but got " << impl_->numAnchors << std::endl;
                        return false;
                    }
                    impl_->numClasses = output_attrs[i].dims[n_dims-3] - (impl_->obj_conf_enabled?5:4);
                }
                else
                {
                    std::cerr << "" << std::endl;
                    return false;
                }
                
            }

            int thisNumDets = impl_->batch * impl_->numAnchors * output_attrs[i].dims[n_dims-2] * output_attrs[i].dims[n_dims-1];
            // actually is start idx
            impl_->all_num_dets.push_back(impl_->numArrays);
            impl_->numArrays += thisNumDets;
            impl_->all_x_grids.push_back(output_attrs[i].dims[n_dims-1]);  // w
            impl_->all_y_grids.push_back(output_attrs[i].dims[n_dims-2]);  // h
        }
    }
    else if (io_num.n_output == strides_.size() * 3)
    {
        impl_->needDecoder = true;
        impl_->splitHead = true;
        for (int i = 0; i < io_num.n_output; i++) 
        {
            output_attrs[i].index = i;
            ret = rknn_query(impl_->ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
            if (impl_->showInfo) dump_tensor_attr(&(output_attrs[i]));
            impl_->output_zps.push_back(output_attrs[i].zp);
            impl_->output_scales.push_back(output_attrs[i].scale);

            if (i >= strides_.size() * 2) 
            {
                int n_dims = output_attrs[i].n_dims; // batch, numArray, h, w
                if(i == strides_.size() * 2)
                {
                    impl_->numClasses = output_attrs[i].dims[n_dims-3] / impl_->numAnchors - (impl_->obj_conf_enabled?5:4);
                }
                int thisNumDets = impl_->batch * impl_->numAnchors * output_attrs[i].dims[n_dims-2] * output_attrs[i].dims[n_dims-1];
                impl_->numArrays += thisNumDets;
                impl_->all_num_dets.push_back(thisNumDets);
                impl_->all_x_grids.push_back(output_attrs[i].dims[n_dims-1]);
                impl_->all_y_grids.push_back(output_attrs[i].dims[n_dims-2]);
            }
        }

    }
    impl_->outputSize = impl_->numArrays * (impl_->numClasses + (impl_->obj_conf_enabled?5:4));
    impl_->inputs.clear();
    impl_->inputs.resize(io_num.n_input);
    for (int i=0;i<io_num.n_input;i++)
    {
        memset(&impl_->inputs[i], 0, sizeof(impl_->inputs[i]));
        impl_->inputs[i].index = i;
        impl_->inputs[i].type = RKNN_TENSOR_UINT8;
        impl_->inputs[i].size = impl_->batch * imgH_ * imgW_ * channel;
        impl_->inputs[i].fmt = impl_->fmt;
        impl_->inputs[i].pass_through = 0;
    }
    
    impl_->outputs.clear();
    impl_->outputs.resize(io_num.n_output);
    for (int i=0;i<io_num.n_output;i++)
    {
        memset(&impl_->outputs[i], 0, sizeof(impl_->outputs[i]));
        impl_->outputs[i].want_float = impl_->needDecoder?impl_->outputFloat:1; // notice this one
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

void YOLO::inference(void* data, void* preds, float scale)
{
    int ret = 0;
    impl_->inputs[0].buf = data;

    // auto t0 = pytime::time();

    ret = rknn_inputs_set(impl_->ctx, 1, impl_->inputs.data());
    if (ret < 0)
    {
        std::cerr << "rknn inputs set error code: " << ret << std::endl;
        return;
    }

    ret = rknn_run(impl_->ctx, NULL);
    if (ret < 0)
    {
        std::cerr << "rknn run error code: " << ret << std::endl;
        return;
    }

    ret = rknn_outputs_get(impl_->ctx, impl_->outputs.size(), impl_->outputs.data(), NULL);
    if (ret < 0)
    {
        std::cerr << "rknn outputs get error code: " << ret << std::endl;
        return;
    }

    // std::cout << "   infer: " << (pytime::time() - t0) * 1000 << "ms, ";
    // t0 = pytime::time();

    // std::cout << (bool)impl_->outputFloat << std::endl;
    // std::cout << (bool)impl_->outputFloat << std::endl;

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
                    impl_->outputs[i + numStrides * 0].buf,
                    impl_->outputs[i + numStrides * 1].buf,
                    impl_->outputs[i + numStrides * 2].buf,
                    impl_->numClasses, true, impl_->outputFloat, impl_->obj_conf_enabled, 
                    impl_->all_x_grids[i], impl_->all_y_grids[i], strides_[i], nowAnchor,
                    impl_->output_zps[i + numStrides * 0], 
                    impl_->output_zps[i + numStrides * 1], 
                    impl_->output_zps[i + numStrides * 2], 
                    impl_->output_scales[i + numStrides * 0], 
                    impl_->output_scales[i + numStrides * 1], 
                    impl_->output_scales[i + numStrides * 2],
                    impl_->regExp, impl_->regMul, 0, impl_->confThres
                );

                delete impl_->outputs[i + numStrides * 0].buf;
                delete impl_->outputs[i + numStrides * 1].buf;
                delete impl_->outputs[i + numStrides * 2].buf;

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
                    impl_->outputs[i].buf, nullptr, nullptr,
                    impl_->numClasses, false, impl_->outputFloat, impl_->obj_conf_enabled, 
                    impl_->all_x_grids[i], impl_->all_y_grids[i], 
                    strides_[i], nowAnchor,
                    impl_->output_zps[i], 0, 0, 
                    impl_->output_scales[i], 0, 0,
                    impl_->regExp, impl_->regMul, 0, impl_->confThres
                );

                delete impl_->outputs[i].buf;

                results.insert(results.end(), result.begin(), result.end());
            }
        }
    }
    else
    {
        generate_yolo_proposals(impl_->numArrays, (float*)impl_->outputs[0].buf, impl_->confThres, results, impl_->numClasses);
        delete impl_->outputs[0].buf;
        // memcpy(preds, impl_->outputs[0].buf, impl_->outputSize * sizeof(float));
    }

    std::vector<int> picked;
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

    // std::cout << "post-process: " << (pytime::time() - t0) * 1000 << "ms, " << std::endl;
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
    delete impl_;
    impl_ = nullptr;
}

#endif
