#ifndef YOLO_PLATFORM_COMMON_HPP
#define YOLO_PLATFORM_COMMON_HPP

#include <iostream>
#include <vector>
#include <math.h>


static bool checkInputSize(int w, int h, int stride=32)
{
    return !(w % stride || h % stride);
}


static int countLengthArray(int w, int h, std::vector<int> strides={8, 16, 32})
{
    if (!checkInputSize(w, h, strides[strides.size()-1]))
    {
        std::cerr << "size not match!" << std::endl;
        exit(-1);
    }
    int ret = 0;
    for (int stride: strides)
    {
        ret += (w / stride) * (h / stride);
    }
    return ret;
}


inline static int32_t __clip(float val, float min, float max) {
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}


static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale) {
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}


static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { 
    return ((float)qnt - (float)zp) * scale; 
}


static float sigmoid(float x) {
    return 1 / (1 + expf(-x));
}


static double unsigmoid(double x) {
    return -logf((1 - x) / x);
}


void qsort_descent_inplace_(std::vector<std::vector<float>>& objects, int left, int right) {
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2][5];

    while (i <= j)
    {
        while (objects[i][5] > p)
            i++;

        while (objects[j][5] < p)
            j--;

        if (i <= j)
        {
            std::swap(objects[i], objects[j]);
            i++;
            j--;
        }
    }

    if (left < j) qsort_descent_inplace_(objects, left, j);
    if (i < right) qsort_descent_inplace_(objects, i, right);

}

void qsort_descent_inplace(std::vector<std::vector<float>>& objects) {
    if (objects.empty())
        return;
    qsort_descent_inplace_(objects, 0, objects.size() - 1);
}


template <class T>
inline T intersection_area(const std::vector<T>& a, const std::vector<T>& b) {
    T x1 = std::max(a[0], b[0]);
    T y1 = std::max(a[1], b[1]);
    T x2 = std::min(a[0] + a[2], b[0] + b[2]);
    T y2 = std::min(a[1] + a[3], b[1] + b[3]);

    if (x1 >= x2 || y1 >= y2) return 0.; 
    return (x2 - x1) * (y2 - y1);
}

template <class T>
inline float IOU(const std::vector<T>& a, const std::vector<T>& b) {
    T inter_area = intersection_area(a, b);
    T union_area = a[2] * a[3] + b[2] * b[3] - inter_area;
    return ((float)inter_area / (float)union_area);
}

template <class T>
void nms_sorted_bboxes(const std::vector<std::vector<T>>& objects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = objects.size();

    std::vector<T> areas(n);
    for (int i = 0; i < n; i++) areas[i] = objects[i][2] * objects[i][3];

    for (int i = 0; i < n; i++) {
        const std::vector<T>& a = objects[i];

        bool keep = true;
        for (int j = 0; j < (int)picked.size(); j++) {
            T inter_area = intersection_area(a, objects[picked[j]]);
            T union_area = areas[i] + areas[picked[j]] - inter_area;

            if ((float)inter_area / (float)union_area > nms_threshold) {
                keep = false;
                break;
            }
        }

        if (keep) picked.push_back(i);
    }
}

static void generate_yolo_proposals(int num_array, float* preds, float prob_threshold, std::vector<std::vector<float>>& objects, int num_class)
{
    for (int anchor_idx = 0; anchor_idx < num_array; anchor_idx++)
    {
        const int basic_pos = anchor_idx * (num_class + 5);

        float box_objectness = preds[basic_pos+4];    // obj conf
        if (box_objectness < prob_threshold) continue;

        int cls_idx=0;
        float max_prob= preds[basic_pos + 5];
        for (int class_idx = 1; class_idx < num_class; class_idx++)
        {
            float box_cls_score = preds[basic_pos + 5 + class_idx];
            if (box_cls_score > max_prob) {
                cls_idx = class_idx;
                max_prob = box_cls_score;
            }
        }

        max_prob *= box_objectness;
        if(max_prob > prob_threshold) {
            std::vector<float> obj;
            obj.resize(6);
            obj[2] = preds[basic_pos+2];
            obj[3] = preds[basic_pos+3];
            obj[0] = preds[basic_pos] - obj[2] * 0.5f;
            obj[1] = preds[basic_pos+1] - obj[3] * 0.5f;
            obj[4] = cls_idx;
            obj[5] = max_prob;
            objects.push_back(obj);
        }
    } // point anchor loop
}


static std::vector<std::vector<float>> decodeOutputs(
    void* reg_, void* conf_, void* cls_, 
    int numClasses, bool splitOutput, bool outputFloat, bool objConfEnabled, int gridX, int gridY, int stride, 
    std::vector<std::vector<float>> anchor, int32_t regZP, int32_t confZP, int32_t clsZP,
    float regScale, float confScale, float clsScale, 
    bool regExp=false, bool regMul=false, int batchID=0, float conf_thres=0.25
    // 目标框使用指数运算， 目标框使用乘法运算
)
{
    std::vector<std::vector<float>> ret;
    // batch * numAnchors * (4 + 1 + numClasses) * w * h
    float unsigmoid_conf_thres = unsigmoid(conf_thres);
    

    int count = 0;
    int numAnchors = anchor.size();
    int numGrids = gridX * gridY;
    // xyxy c c1-cn
    int lengthArray = 5+numClasses;
    
    int offset = 0, bias = 0, predsBias = 0;
    int regBatchBias=0, confBatchBias=0, clsBatchBias=0;

    // std::cout << "outputFloat: " << outputFloat << std::endl;

    if(outputFloat)
    {
        float* reg = (float*)reg_;
        float* conf = nullptr;
        float* cls = nullptr;
        float nowObjectness = 0., nowClassConf=0.;

        if (splitOutput)
        {
            regBatchBias = batchID * numAnchors * 4 * numGrids;
            clsBatchBias = batchID * numAnchors * numClasses * numGrids;
            if(objConfEnabled)
            {
                conf = (float*)conf_;
                confBatchBias = batchID * numAnchors * numGrids;
            }
            cls = (float*)cls_;

            for (int i=0;i<gridY;i++)
            {
                offset = i * gridX;
                for (int j=0;j<gridX;j++)
                {
                    bias = offset + j;
                    predsBias = bias * lengthArray * numAnchors;
                    for (int iA=0;iA<numAnchors;iA++)
                    {
                        // conf
                        if (objConfEnabled)
                        {
                            nowObjectness = conf[confBatchBias + iA * numGrids + bias];
                        }
                        else
                        {
                            nowObjectness = 1.0;
                        }
                        if (regExp || regMul)
                        {
                            if (nowObjectness < unsigmoid_conf_thres) continue;
                            if (objConfEnabled) nowObjectness = sigmoid(nowObjectness);

                            int maxK=-1;
                            float maxConf=conf_thres;
                            for (int k=0;k<numClasses;k++)
                            {
                                nowClassConf = nowObjectness * sigmoid(cls[clsBatchBias + (iA * numClasses + k) * numGrids + bias]);
                                if (nowClassConf >= maxConf)
                                {
                                    maxK = k;
                                    maxConf = nowClassConf;
                                }
                            }
                            if (maxK >= 0)
                            {
                                std::vector<float> obj;
                                obj.resize(6);
                                obj[0] = reg[regBatchBias + (iA * 4 /*+0*/) * numGrids + bias];
                                obj[1] = reg[regBatchBias + (iA * 4 + 1) * numGrids + bias];
                                obj[2] = reg[regBatchBias + (iA * 4 + 2) * numGrids + bias];
                                obj[3] = reg[regBatchBias + (iA * 4 + 3) * numGrids + bias];

                                if (regExp)
                                {
                                    obj[2] = expf(obj[2]) * anchor[iA][0] * stride;   // w
                                    obj[3] = expf(obj[3]) * anchor[iA][1] * stride;   // h
                                    obj[0] = (obj[0] + j) * stride - obj[2] * 0.5f;   // x
                                    obj[1] = (obj[1] + i) * stride - obj[3] * 0.5f;   // y
                                }
                                else if (regMul)
                                {
                                    obj[0] = (sigmoid(obj[0]) * 2. + j - 0.5) * stride;
                                    obj[1] = (sigmoid(obj[1]) * 2. + i - 0.5) * stride;

                                    obj[2] = pow(sigmoid(obj[2]) * 2, 2) * anchor[iA][0] * stride;
                                    obj[3] = pow(sigmoid(obj[3]) * 2, 2) * anchor[iA][1] * stride;
                                }
                                obj[4] = maxK;
                                obj[5] = maxConf;
                                ret.push_back(obj);
                            }
                        }
                        else
                        {
                            if (nowObjectness < conf_thres) continue;
                            
                            int maxK=-1;
                            float maxConf=conf_thres;
                            for (int k=0;k<numClasses;k++)
                            {
                                nowClassConf = nowObjectness * cls[clsBatchBias + (iA * numClasses + k) * numGrids + bias];
                                if (nowClassConf >= maxConf)
                                {
                                    maxK = k;
                                    maxConf = nowClassConf;
                                }
                            }
                            if (maxK >= 0)
                            {
                                std::vector<float> obj;
                                obj.resize(6);
                                obj[0] = reg[regBatchBias + (iA * 4 /*+0*/) * numGrids + bias];
                                obj[1] = reg[regBatchBias + (iA * 4 + 1) * numGrids + bias];
                                obj[2] = reg[regBatchBias + (iA * 4 + 2) * numGrids + bias];
                                obj[3] = reg[regBatchBias + (iA * 4 + 3) * numGrids + bias];
                                obj[4] = maxK;
                                obj[5] = maxConf;
                                ret.push_back(obj);
                            }
                        }
                        

                    }
                }
            }
        }
        else
        {
            regBatchBias = batchID * numAnchors * lengthArray * numGrids;
            int lengthNotClass = objConfEnabled?5:4;
            int regArrayLength = numClasses + lengthNotClass;
            for (int i=0;i<gridY;i++)
            {
                offset = i * gridX;
                for (int j=0;j<gridX;j++)
                {
                    bias = offset + j;
                    predsBias = bias * lengthArray * numAnchors;
                    for (int iA=0;iA<numAnchors;iA++)
                    {
                        // conf
                        nowObjectness = objConfEnabled?reg[regBatchBias + (iA * regArrayLength + 4) * numGrids + bias]:1;

                        if (regExp || regMul)
                        {
                            if (nowObjectness < unsigmoid_conf_thres) continue;
                            if (objConfEnabled) nowObjectness = sigmoid(nowObjectness);

                            int maxK=-1;
                            float maxConf=conf_thres;
                            for (int k=0;k<numClasses;k++)
                            {
                                nowClassConf = nowObjectness * sigmoid(reg[regBatchBias + (iA * regArrayLength + lengthNotClass + k) * numGrids + bias]);
                                if (nowClassConf >= maxConf)
                                {
                                    maxK = k;
                                    maxConf = nowClassConf;
                                }
                            }
                            if (maxK >= 0)
                            {
                                std::vector<float> obj;
                                obj.resize(6);
                                obj[0] = reg[regBatchBias + (iA * regArrayLength /*+0*/) * numGrids + bias];
                                obj[1] = reg[regBatchBias + (iA * regArrayLength + 1) * numGrids + bias];
                                obj[2] = reg[regBatchBias + (iA * regArrayLength + 2) * numGrids + bias];
                                obj[3] = reg[regBatchBias + (iA * regArrayLength + 3) * numGrids + bias];

                                if (regExp)
                                {
                                    obj[2] = expf(obj[2]) * anchor[iA][0] * stride;   // w
                                    obj[3] = expf(obj[3]) * anchor[iA][1] * stride;   // h
                                    obj[0] = (obj[0] + j) * stride - obj[2] * 0.5f;   // x
                                    obj[1] = (obj[1] + i) * stride - obj[3] * 0.5f;   // y
                                }
                                else if (regMul)
                                {
                                    obj[0] = (sigmoid(obj[0]) * 2. + j - 0.5) * stride;
                                    obj[1] = (sigmoid(obj[1]) * 2. + i - 0.5) * stride;

                                    obj[2] = pow(sigmoid(obj[2]) * 2, 2) * anchor[iA][0] * stride;
                                    obj[3] = pow(sigmoid(obj[3]) * 2, 2) * anchor[iA][1] * stride;
                                }
                                obj[4] = maxK;
                                obj[5] = maxConf;
                                ret.push_back(obj);
                            }

                        }
                        else
                        {
                            if (nowObjectness < conf_thres) continue;
                            
                            int maxK=-1;
                            float maxConf=conf_thres;
                            for (int k=0;k<numClasses;k++)
                            {
                                nowClassConf = nowObjectness * reg[regBatchBias + (iA * regArrayLength + lengthNotClass + k) * numGrids + bias];
                                if (nowClassConf >= maxConf)
                                {
                                    maxK = k;
                                    maxConf = nowClassConf;
                                }
                            }
                            if (maxK >= 0)
                            {
                                std::vector<float> obj;
                                obj.resize(6);
                                obj[0] = reg[regBatchBias + (iA * regArrayLength /*+0*/) * numGrids + bias];
                                obj[1] = reg[regBatchBias + (iA * regArrayLength + 1) * numGrids + bias];
                                obj[2] = reg[regBatchBias + (iA * regArrayLength + 2) * numGrids + bias];
                                obj[3] = reg[regBatchBias + (iA * regArrayLength + 3) * numGrids + bias];
                                obj[4] = maxK;
                                obj[5] = maxConf;
                                ret.push_back(obj);
                            }
                        }
                    }
                }
            }

        }
    
    }
    else
    {
        int8_t* reg = (int8_t*)reg_;
        int8_t* conf = nullptr;
        int8_t* cls = nullptr;
        if (!splitOutput)
        {
            confZP = regZP;
            confScale = regScale;
            clsZP = regZP;
            clsScale = regScale;
        }
        int8_t qnt_unsigmoid_conf_thres = qnt_f32_to_affine(unsigmoid_conf_thres, confZP, confScale);

        if (splitOutput)
        {
            regBatchBias = batchID * numAnchors * 4 * numGrids;
            clsBatchBias = batchID * numAnchors * numClasses * numGrids;
            float nowObjectness = 0., nowClassConf=0.;
            if(objConfEnabled)
            {
                conf = (int8_t*)conf_;
                confBatchBias = batchID * numAnchors * numGrids;
            }
            cls = (int8_t*)cls_;

            for (int i=0;i<gridY;i++)
            {
                offset = i * gridX;
                for (int j=0;j<gridX;j++)
                {
                    bias = offset + j;
                    predsBias = bias * lengthArray * numAnchors;
                    for (int iA=0;iA<numAnchors;iA++)
                    {
                        // conf
                        if (objConfEnabled)
                        {
                            if (conf[confBatchBias + iA * numGrids + bias] < qnt_unsigmoid_conf_thres) continue;
                            // std::cout << "conf accpet at " << i << ", " << j << ", " << iA << std::endl;
                            nowObjectness = deqnt_affine_to_f32(conf[confBatchBias + iA * numGrids + bias], confZP, confScale);
                        }
                        else
                        {
                            nowObjectness = 1.0;
                        }

                        if (regExp || regMul)
                        {
                            if (objConfEnabled) nowObjectness = sigmoid(nowObjectness);

                            int maxK=-1;
                            float maxConf=conf_thres;
                            for (int k=0;k<numClasses;k++)
                            {
                                nowClassConf = nowObjectness * sigmoid(deqnt_affine_to_f32(cls[clsBatchBias + (iA * numClasses + k) * numGrids + bias], clsZP, clsScale));
                                if (nowClassConf >= maxConf)
                                {
                                    maxK = k;
                                    maxConf = nowClassConf;
                                }
                            }
                            if (maxK >= 0)
                            {
                                std::vector<float> obj;
                                obj.resize(6);
                                obj[0] = deqnt_affine_to_f32(reg[regBatchBias + (iA * 4 /*+0*/) * numGrids + bias], regZP, regScale);
                                obj[1] = deqnt_affine_to_f32(reg[regBatchBias + (iA * 4 + 1) * numGrids + bias], regZP, regScale);
                                obj[2] = deqnt_affine_to_f32(reg[regBatchBias + (iA * 4 + 2) * numGrids + bias], regZP, regScale);
                                obj[3] = deqnt_affine_to_f32(reg[regBatchBias + (iA * 4 + 3) * numGrids + bias], regZP, regScale);

                                if (regExp)
                                {
                                    // obj[0] = (obj[0] + j) * stride;   // x
                                    // obj[1] = (obj[1] + i) * stride;   // y

                                    obj[2] = expf(obj[2]) * anchor[iA][0] * stride;   // w
                                    obj[3] = expf(obj[3]) * anchor[iA][1] * stride;   // h
                                    obj[0] = (obj[0] + j) * stride - obj[2] * 0.5f;   // x
                                    obj[1] = (obj[1] + i) * stride - obj[3] * 0.5f;   // y
                                }
                                else if (regMul)
                                {
                                    obj[0] = (sigmoid(obj[0]) * 2. + j - 0.5) * stride;
                                    obj[1] = (sigmoid(obj[1]) * 2. + i - 0.5) * stride;

                                    obj[2] = pow(sigmoid(obj[2]) * 2, 2) * anchor[iA][0] * stride;
                                    obj[3] = pow(sigmoid(obj[3]) * 2, 2) * anchor[iA][1] * stride;
                                }
                                obj[4] = maxK;
                                obj[5] = maxConf;
                                ret.push_back(obj);
                            }
                        }
                        else
                        {
                            if (nowObjectness < conf_thres) continue;
                            
                            int maxK=-1;
                            float maxConf=conf_thres;
                            for (int k=0;k<numClasses;k++)
                            {
                                nowClassConf = nowObjectness * deqnt_affine_to_f32(cls[clsBatchBias + (iA * numClasses + k) * numGrids + bias], clsZP, clsScale);
                                if (nowClassConf >= maxConf)
                                {
                                    maxK = k;
                                    maxConf = nowClassConf;
                                }
                            }
                            if (maxK >= 0)
                            {
                                std::vector<float> obj;
                                obj.resize(6);
                                obj[0] = deqnt_affine_to_f32(reg[regBatchBias + (iA * 4 /*+0*/) * numGrids + bias], regZP, regScale);
                                obj[1] = deqnt_affine_to_f32(reg[regBatchBias + (iA * 4 + 1) * numGrids + bias], regZP, regScale);
                                obj[2] = deqnt_affine_to_f32(reg[regBatchBias + (iA * 4 + 2) * numGrids + bias], regZP, regScale);
                                obj[3] = deqnt_affine_to_f32(reg[regBatchBias + (iA * 4 + 3) * numGrids + bias], regZP, regScale);
                                obj[4] = maxK;
                                obj[5] = maxConf;
                                ret.push_back(obj);
                            }
                        }
                        
                    }
                }
            }
        }
        else
        {
            regBatchBias = batchID * numAnchors * lengthArray * numGrids;
            int lengthNotClass = objConfEnabled?5:4;
            int regArrayLength = numClasses + lengthNotClass;
            float nowObjectness = 0., nowClassConf=0.;
            for (int i=0;i<gridY;i++)
            {
                offset = i * gridX;
                for (int j=0;j<gridX;j++)
                {
                    bias = offset + j;
                    predsBias = bias * lengthArray * numAnchors;
                    for (int iA=0;iA<numAnchors;iA++)
                    {
                        // conf
                        if (objConfEnabled)
                        {
                            if (conf[confBatchBias + iA * numGrids + bias] < qnt_unsigmoid_conf_thres) continue;
                            nowObjectness = deqnt_affine_to_f32(reg[regBatchBias + (iA * regArrayLength + 4) * numGrids + bias], regZP, regScale);
                        }
                        else
                        {
                            nowObjectness = 1.0;
                        }

                        if (regExp || regMul)
                        {
                            if (objConfEnabled) nowObjectness = sigmoid(nowObjectness);

                            int maxK=-1;
                            float maxConf=conf_thres;
                            for (int k=0;k<numClasses;k++)
                            {
                                nowClassConf = nowObjectness * sigmoid(
                                    deqnt_affine_to_f32(
                                        reg[regBatchBias + (iA * regArrayLength + lengthNotClass + k) * numGrids + bias], regZP, regScale
                                    )
                                );

                                if (nowClassConf >= maxConf)
                                {
                                    maxK = k;
                                    maxConf = nowClassConf;
                                }
                            }
                            if (maxK >= 0)
                            {
                                std::vector<float> obj;
                                obj.resize(6);
                                obj[0] = deqnt_affine_to_f32(reg[regBatchBias + (iA * regArrayLength /*+0*/) * numGrids + bias], regZP, regScale);
                                obj[1] = deqnt_affine_to_f32(reg[regBatchBias + (iA * regArrayLength + 1) * numGrids + bias], regZP, regScale);
                                obj[2] = deqnt_affine_to_f32(reg[regBatchBias + (iA * regArrayLength + 2) * numGrids + bias], regZP, regScale);
                                obj[3] = deqnt_affine_to_f32(reg[regBatchBias + (iA * regArrayLength + 3) * numGrids + bias], regZP, regScale);

                                if (regExp)
                                {
                                    obj[0] = (obj[0] + j) * stride;   // x
                                    obj[1] = (obj[1] + i) * stride;   // y

                                    obj[2] = expf(obj[2]) * anchor[iA][0] * stride;   // w
                                    obj[3] = expf(obj[3]) * anchor[iA][1] * stride;   // h
                                }
                                else if (regMul)
                                {
                                    obj[0] = (sigmoid(obj[0]) * 2. + j - 0.5) * stride;
                                    obj[1] = (sigmoid(obj[1]) * 2. + i - 0.5) * stride;

                                    obj[2] = pow(sigmoid(obj[2]) * 2, 2) * anchor[iA][0] * stride;
                                    obj[3] = pow(sigmoid(obj[3]) * 2, 2) * anchor[iA][1] * stride;
                                }
                                obj[4] = maxK;
                                obj[5] = maxConf;
                                ret.push_back(obj);
                            }

                        }
                        else
                        {
                            if (nowObjectness < conf_thres) continue;
                            
                            int maxK=-1;
                            float maxConf=conf_thres;
                            for (int k=0;k<numClasses;k++)
                            {
                                nowClassConf = nowObjectness * deqnt_affine_to_f32(reg[regBatchBias + (iA * regArrayLength + lengthNotClass + k) * numGrids + bias], regZP, regScale);
                                if (nowClassConf >= maxConf)
                                {
                                    maxK = k;
                                    maxConf = nowClassConf;
                                }
                            }
                            if (maxK >= 0)
                            {
                                std::vector<float> obj;
                                obj.resize(6);
                                obj[0] = deqnt_affine_to_f32(reg[regBatchBias + (iA * regArrayLength /*+0*/) * numGrids + bias], regZP, regScale);
                                obj[1] = deqnt_affine_to_f32(reg[regBatchBias + (iA * regArrayLength + 1) * numGrids + bias], regZP, regScale);
                                obj[2] = deqnt_affine_to_f32(reg[regBatchBias + (iA * regArrayLength + 2) * numGrids + bias], regZP, regScale);
                                obj[3] = deqnt_affine_to_f32(reg[regBatchBias + (iA * regArrayLength + 3) * numGrids + bias], regZP, regScale);
                                obj[4] = maxK;
                                obj[5] = maxConf;
                                ret.push_back(obj);
                            }
                        }
                    }
                }
            }

        }
        




    }

    return ret;
}




#endif