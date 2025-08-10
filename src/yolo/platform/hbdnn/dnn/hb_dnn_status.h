// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef DNN_HB_DNN_STATUS_H_
#define DNN_HB_DNN_STATUS_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum {
  HB_DNN_SUCCESS = 0,
  HB_DNN_INVALID_ARGUMENT = -6000001,
  HB_DNN_INVALID_MODEL = -6000002,
  HB_DNN_MODEL_NUMBER_EXCEED_LIMIT = -6000003,
  HB_DNN_INVALID_PACKED_DNN_HANDLE = -6000004,
  HB_DNN_INVALID_DNN_HANDLE = -6000005,
  HB_DNN_CAN_NOT_OPEN_FILE = -6000006,
  HB_DNN_OUT_OF_MEMORY = -6000007,
  HB_DNN_TIMEOUT = -6000008,
  HB_DNN_TASK_NUM_EXCEED_LIMIT = -6000009,
  HB_DNN_TASK_BATCH_SIZE_EXCEED_LIMIT = -6000010,
  HB_DNN_INVALID_TASK_HANDLE = -6000011,
  HB_DNN_RUN_TASK_FAILED = -6000012,
  HB_DNN_MODEL_IS_RUNNING = -6000013,
  HB_DNN_INCOMPATIBLE_MODEL = -6000014,
  HB_DNN_API_USE_ERROR = -6000015,
  HB_DNN_MULTI_PROGRESS_USE_ERROR = -6000016
} hbDNNStatus;

/**
 * Get DNN error code description
 * @param[in] errorCode, dnn error code
 * @return DNN error code description in nature language
 */
char const *hbDNNGetErrorDesc(int32_t errorCode);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // DNN_HB_DNN_STATUS_H_
