/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef INC_EXTERNAL_GE_GE_ERROR_CODES_H_
#define INC_EXTERNAL_GE_GE_ERROR_CODES_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif
static const uint32_t ACL_ERROR_GE_PARAM_INVALID = 145000;
static const uint32_t ACL_ERROR_GE_EXEC_NOT_INIT = 145001;
static const uint32_t ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID = 145002;
static const uint32_t ACL_ERROR_GE_EXEC_MODEL_ID_INVALID = 145003;
static const uint32_t ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID = 145006;
static const uint32_t ACL_ERROR_GE_EXEC_MODEL_ADDR_INVALID = 145007;
static const uint32_t ACL_ERROR_GE_EXEC_MODEL_QUEUE_ID_INVALID = 145008;
static const uint32_t ACL_ERROR_GE_EXEC_LOAD_MODEL_REPEATED = 145009;
static const uint32_t ACL_ERROR_GE_DYNAMIC_INPUT_ADDR_INVALID = 145011;
static const uint32_t ACL_ERROR_GE_DYNAMIC_INPUT_LENGTH_INVALID = 145012;
static const uint32_t ACL_ERROR_GE_DYNAMIC_BATCH_SIZE_INVALID = 145013;
static const uint32_t ACL_ERROR_GE_AIPP_BATCH_EMPTY = 145014;
static const uint32_t ACL_ERROR_GE_AIPP_NOT_EXIST = 145015;
static const uint32_t ACL_ERROR_GE_AIPP_MODE_INVALID = 145016;
static const uint32_t ACL_ERROR_GE_OP_TASK_TYPE_INVALID = 145017;
static const uint32_t ACL_ERROR_GE_OP_KERNEL_TYPE_INVALID = 145018;
static const uint32_t ACL_ERROR_GE_MEMORY_ALLOCATION = 245000;
static const uint32_t ACL_ERROR_GE_INTERNAL_ERROR = 545000;
static const uint32_t ACL_ERROR_GE_LOAD_MODEL = 545001;
static const uint32_t ACL_ERROR_GE_EXEC_LOAD_MODEL_PARTITION_FAILED = 545002;
static const uint32_t ACL_ERROR_GE_EXEC_LOAD_WEIGHT_PARTITION_FAILED = 545003;
static const uint32_t ACL_ERROR_GE_EXEC_LOAD_TASK_PARTITION_FAILED = 545004;
static const uint32_t ACL_ERROR_GE_EXEC_LOAD_KERNEL_PARTITION_FAILED = 545005;
static const uint32_t ACL_ERROR_GE_EXEC_RELEASE_MODEL_DATA = 545006;
static const uint32_t ACL_ERROR_GE_COMMAND_HANDLE = 545007;
static const uint32_t ACL_ERROR_GE_GET_TENSOR_INFO = 545008;
static const uint32_t ACL_ERROR_GE_UNLOAD_MODEL = 545009;
#ifdef __cplusplus
}  // namespace ge
#endif
#endif  // INC_EXTERNAL_GE_GE_ERROR_CODES_H_
