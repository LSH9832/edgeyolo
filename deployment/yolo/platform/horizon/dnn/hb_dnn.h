// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef DNN_HB_DNN_H_
#define DNN_HB_DNN_H_

#include "./hb_sys.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define HB_DNN_VERSION_MAJOR 1U
#define HB_DNN_VERSION_MINOR 22U
#define HB_DNN_VERSION_PATCH 6U

#define HB_DNN_TENSOR_MAX_DIMENSIONS 8

#define HB_DNN_INITIALIZE_INFER_CTRL_PARAM(param) \
  {                                               \
    (param)->bpuCoreId = HB_BPU_CORE_ANY;         \
    (param)->dspCoreId = HB_DSP_CORE_ANY;         \
    (param)->priority = HB_DNN_PRIORITY_LOWEST;   \
    (param)->more = false;                        \
    (param)->customId = 0;                        \
    (param)->reserved1 = 0;                       \
    (param)->reserved2 = 0;                       \
  }

typedef void *hbPackedDNNHandle_t;
typedef void *hbDNNHandle_t;
typedef void *hbDNNTaskHandle_t;

typedef enum {
  HB_DNN_LAYOUT_NHWC = 0,
  HB_DNN_LAYOUT_NCHW = 2,
  HB_DNN_LAYOUT_NONE = 255,
} hbDNNTensorLayout;

typedef enum {
  HB_DNN_IMG_TYPE_Y,
  HB_DNN_IMG_TYPE_NV12,
  HB_DNN_IMG_TYPE_NV12_SEPARATE,
  HB_DNN_IMG_TYPE_YUV444,
  HB_DNN_IMG_TYPE_RGB,
  HB_DNN_IMG_TYPE_BGR,
  HB_DNN_TENSOR_TYPE_S4,
  HB_DNN_TENSOR_TYPE_U4,
  HB_DNN_TENSOR_TYPE_S8,
  HB_DNN_TENSOR_TYPE_U8,
  HB_DNN_TENSOR_TYPE_F16,
  HB_DNN_TENSOR_TYPE_S16,
  HB_DNN_TENSOR_TYPE_U16,
  HB_DNN_TENSOR_TYPE_F32,
  HB_DNN_TENSOR_TYPE_S32,
  HB_DNN_TENSOR_TYPE_U32,
  HB_DNN_TENSOR_TYPE_F64,
  HB_DNN_TENSOR_TYPE_S64,
  HB_DNN_TENSOR_TYPE_U64,
  HB_DNN_TENSOR_TYPE_MAX
} hbDNNDataType;

typedef struct {
  int32_t dimensionSize[HB_DNN_TENSOR_MAX_DIMENSIONS];
  int32_t numDimensions;
} hbDNNTensorShape;

typedef struct {
  int32_t shiftLen;
  uint8_t *shiftData;
} hbDNNQuantiShift;

/**
 * Quantize/Dequantize by scale
 * For Dequantize:
 * if zeroPointLen = 0 f(x_i) = x_i * scaleData[i]
 * if zeroPointLen > 0 f(x_i) = (x_i - zeroPointData[i]) * scaleData[i]
 * 
 * For Quantize:
 * if zeroPointLen = 0 f(x_i) = g(x_i / scaleData[i])
 * if zeroPointLen > 0 f(x_i) = g(x_i / scaleData[i] + zeroPointData[i])
 * which g(x) = clip(nearbyint(x)), use fesetround(FE_TONEAREST), U8: 0 <= g(x) <= 255, S8: -128 <= g(x) <= 127
 */
typedef struct {
  int32_t scaleLen;
  float *scaleData;
  int32_t zeroPointLen;
  int8_t *zeroPointData;
} hbDNNQuantiScale;

typedef enum {
  NONE,  // no quantization
  SHIFT,
  SCALE
} hbDNNQuantiType;

typedef struct {
  hbDNNTensorShape validShape;
  hbDNNTensorShape alignedShape;
  int32_t tensorLayout;
  int32_t tensorType;
  hbDNNQuantiShift shift;
  hbDNNQuantiScale scale;
  hbDNNQuantiType quantiType;
  int32_t quantizeAxis;
  int32_t alignedByteSize;
  int32_t stride[HB_DNN_TENSOR_MAX_DIMENSIONS];
} hbDNNTensorProperties;

typedef struct {
  hbSysMem sysMem[4];
  hbDNNTensorProperties properties;
} hbDNNTensor;

typedef struct {
  int32_t left;
  int32_t top;
  int32_t right;
  int32_t bottom;
} hbDNNRoi;

typedef enum {
  HB_DNN_PRIORITY_LOWEST = 0,
  HB_DNN_PRIORITY_HIGHEST = 255,
  HB_DNN_PRIORITY_PREEMP = HB_DNN_PRIORITY_HIGHEST,
} hbDNNTaskPriority;

typedef struct {
  int32_t bpuCoreId;
  int32_t dspCoreId;
  int32_t priority;
  int32_t more;
  int64_t customId;
  int32_t reserved1;
  int32_t reserved2;
} hbDNNInferCtrlParam;

typedef void (*hbDNNTaskDoneCb)(hbDNNTaskHandle_t taskHandle, int32_t status,
                                void *userdata);

/**
 * Get DNN version
 * @return DNN version info
 */
char const *hbDNNGetVersion();

/**
 * Creates and initializes Horizon DNN Networks from file list
 * @param[out] packedDNNHandle
 * @param[in] modelFileNames
 * @param[in] modelFileCount
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNInitializeFromFiles(hbPackedDNNHandle_t *packedDNNHandle,
                                 char const **modelFileNames,
                                 int32_t modelFileCount);

/**
 * Creates and initializes Horizon DNN Networks from memory
 * @param[out] packedDNNHandle
 * @param[in] modelData
 * @param[in] modelDataLengths
 * @param[in] modelDataCount
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNInitializeFromDDR(hbPackedDNNHandle_t *packedDNNHandle,
                               const void **modelData,
                               int32_t *modelDataLengths,
                               int32_t modelDataCount);

/**
 * Release DNN Networks in a given packed handle
 * @param[in] packedDNNHandle
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNRelease(hbPackedDNNHandle_t packedDNNHandle);

/**
 * Get model names from given packed handle
 * @param[out] modelNameList
 * @param[out] modelNameCount
 * @param[in] packedDNNHandle
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNGetModelNameList(char const ***modelNameList,
                              int32_t *modelNameCount,
                              hbPackedDNNHandle_t packedDNNHandle);

/**
 * Get DNN Network handle from packed Handle with given model name
 * @param[out] dnnHandle
 * @param[in] packedDNNHandle
 * @param[in] modelName
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNGetModelHandle(hbDNNHandle_t *dnnHandle,
                            hbPackedDNNHandle_t packedDNNHandle,
                            char const *modelName);

/**
 * Get input count
 * @param[out] inputCount
 * @param[in] dnnHandle
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNGetInputCount(int32_t *inputCount, hbDNNHandle_t dnnHandle);

/**
 * Get model input name
 * @param[out] name
 * @param[in] dnnHandle
 * @param[in] inputIndex
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNGetInputName(char const **name, hbDNNHandle_t dnnHandle,
                          int32_t inputIndex);

/**
 * Get input tensor properties
 * @param[out] properties
 * @param[in] dnnHandle
 * @param[in] inputIndex
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNGetInputTensorProperties(hbDNNTensorProperties *properties,
                                      hbDNNHandle_t dnnHandle,
                                      int32_t inputIndex);

/**
 * Get output count
 * @param[out] outputCount
 * @param[in] dnnHandle
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNGetOutputCount(int32_t *outputCount, hbDNNHandle_t dnnHandle);

/**
 * Get model output name
 * @param[out] name
 * @param[in] dnnHandle
 * @param[in] outputIndex
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNGetOutputName(char const **name, hbDNNHandle_t dnnHandle,
                           int32_t outputIndex);

/**
 * Get output tensor properties
 * @param[out] properties
 * @param[in] dnnHandle
 * @param[in] outputIndex
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNGetOutputTensorProperties(hbDNNTensorProperties *properties,
                                       hbDNNHandle_t dnnHandle,
                                       int32_t outputIndex);

/**
 * DNN inference
 * @param[out] taskHandle: return a pointer represent the task if success,  otherwise nullptr
 * @param[out] output: pointer to the output tensor array, the size of array should be equal to $(`hbDNNGetOutputCount`)
 * @param[in] input: input tensor array, the size of array should be equal to  $(`hbDNNGetInputCount`)
 * @param[in] dnnHandle: pointer to the dnn handle
 * @param[in] inferCtrlParam: infer control parameters
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNInfer(hbDNNTaskHandle_t *taskHandle, hbDNNTensor **output,
                   hbDNNTensor const *input, hbDNNHandle_t dnnHandle,
                   hbDNNInferCtrlParam *inferCtrlParam);

/**
 * DNN inference with rois
 * @param[out] taskHandle: return a pointer represent the task if success,  otherwise nullptr
 * @param[out] output: pointer to the output tensor array
 * @param[in] input: input tensor array, the size of array should be equal to  $(`hbDNNGetInputCount`) * `batch`
 *      range of [idx*$(`hbDNNGetInputCount`), (idx+1)*$(`hbDNNGetInputCount`)) represents input tensors
 *      for idxth batch. 
 * @param[in] rois: Rois. the size of array should be equal to roiCount. 
 *      Assuming that the model has the input of n resizer input sources, range of [idx*n, (idx+1)*n) represents 
 *      rois for idxth batch.
 * @param[in] roiCount: roi count. If the model has n resizer input sources, then roiCount=`batch` * n.
 * @param[in] dnnHandle: pointer to the dnn handle
 * @param[in] inferCtrlParam: infer control parameters
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNRoiInfer(hbDNNTaskHandle_t *taskHandle, hbDNNTensor **output,
                      hbDNNTensor const *input, hbDNNRoi *rois,
                      int32_t roiCount, hbDNNHandle_t dnnHandle,
                      hbDNNInferCtrlParam *inferCtrlParam);

/**
 * DNN set task done callback
 * @param[in] taskHandle: pointer to the task
 * @param[in] cb: callback function
 * @param[in] userdata: userdata 
 * @return 0 if success, return defined error code otherwise
*/
int32_t hbDNNSetTaskDoneCb(hbDNNTaskHandle_t taskHandle, hbDNNTaskDoneCb cb,
                           void *userdata);

/**
 * Wait util task completed or timeout.
 * @param[in] taskHandle: pointer to the task
 * @param[in] timeout: timeout of milliseconds
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNWaitTaskDone(hbDNNTaskHandle_t taskHandle, int32_t timeout);

/**
 * Release a task and its related resources. If the task has not been executed then it will be canceled,
 * and if the task has not been finished then it will be stopped.
 * This interface will return immediately, and all operations will run in the background
 * @param[in] taskHandle: pointer to the task
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNReleaseTask(hbDNNTaskHandle_t taskHandle);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // DNN_HB_DNN_H_
