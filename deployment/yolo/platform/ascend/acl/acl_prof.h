/**
* @file acl_prof.h
*
* Copyright (C) Huawei Technologies Co., Ltd. 2019-2020. All Rights Reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#ifndef INC_EXTERNAL_ACL_PROF_H_
#define INC_EXTERNAL_ACL_PROF_H_

#include "./acl_base.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ACL_PROF_ACL_API                0x0001
#define ACL_PROF_TASK_TIME              0x0002
#define ACL_PROF_AICORE_METRICS         0x0004
#define ACL_PROF_AICPU                  0x0008

#define ACL_PROF_MAX_OP_NAME_LEN        257
#define ACL_PROF_MAX_OP_TYPE_LEN        65

typedef enum {
    ACL_AICORE_ARITHMETIC_UTILIZATION = 0,
    ACL_AICORE_PIPE_UTILIZATION = 1,
    ACL_AICORE_MEMORY_BANDWIDTH = 2,
    ACL_AICORE_L0B_AND_WIDTH = 3,
    ACL_AICORE_RESOURCE_CONFLICT_RATIO = 4,
    ACL_AICORE_NONE = 0xFF
} aclprofAicoreMetrics;

typedef struct aclprofConfig aclprofConfig;
typedef struct aclprofStopConfig aclprofStopConfig;
typedef struct aclprofAicoreEvents aclprofAicoreEvents;
typedef struct aclprofSubscribeConfig aclprofSubscribeConfig;

/**
 * @ingroup AscendCL
 * @brief profiling initialize
 *
 * @param  profilerResultPath [IN]  path of profiling result
 * @param  length [IN]              length of profilerResultPath
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclprofFinalize
 */
ACL_FUNC_VISIBILITY aclError aclprofInit(const char *profilerResultPath, size_t length);

/**
 * @ingroup AscendCL
 * @brief profiling finalize
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclprofInit
 */
ACL_FUNC_VISIBILITY aclError aclprofFinalize();

/**
 * @ingroup AscendCL
 * @brief Start profiling modules by profilerConfig
 *
 * @param  profilerConfig [IN]  config of profiling
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclprofStop
 */
ACL_FUNC_VISIBILITY aclError aclprofStart(const aclprofConfig *profilerConfig);

/**
 * @ingroup AscendCL
 * @brief Create data of type aclprofConfig
 *
 * @param  deviceIdList [IN]      list of device id
 * @param  deviceNums [IN]        number of devices
 * @param  aicoreMetrics [IN]     type of aicore metrics
 * @param  aicoreEvents [IN]      pointer to aicore events, only support NULL now
 * @param  dataTypeConfig [IN]    config modules need profiling
 *
 * @retval the aclprofConfig pointer
 *
 * @see aclprofDestroyConfig
 */
ACL_FUNC_VISIBILITY aclprofConfig *aclprofCreateConfig(uint32_t *deviceIdList, uint32_t deviceNums,
    aclprofAicoreMetrics aicoreMetrics, aclprofAicoreEvents *aicoreEvents, uint64_t dataTypeConfig);

/**
 * @ingroup AscendCL
 * @brief Destroy data of type aclprofConfig
 *
 * @param  profilerConfig [IN]  config of profiling
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclprofCreateConfig
 */
ACL_FUNC_VISIBILITY aclError aclprofDestroyConfig(const aclprofConfig *profilerConfig);

/**
 * @ingroup AscendCL
 * @brief stop profiling modules by stopProfilingConfig
 *
 * @param  profilerConfig [IN]  pointer to stop config of profiling
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclprofStart
 */
ACL_FUNC_VISIBILITY aclError aclprofStop(const aclprofConfig *profilerConfig);

/**
 * @ingroup AscendCL
 * @brief subscribe profiling data of model
 *
 * @param  modelId [IN]              the model id subscribed
 * @param  profSubscribeConfig [IN]  pointer to config of model subscribe
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclprofModelUnSubscribe
 */
ACL_FUNC_VISIBILITY aclError aclprofModelSubscribe(uint32_t modelId,
    const aclprofSubscribeConfig *profSubscribeConfig);

/**
 * @ingroup AscendCL
 * @brief unsubscribe profiling data of model
 *
 * @param  modelId [IN]  the model id unsubscribed
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclprofModelSubscribe
 */
ACL_FUNC_VISIBILITY aclError aclprofModelUnSubscribe(uint32_t modelId);

/**
 * @ingroup AscendCL
 * @brief create subscribe config
 *
 * @param  timeInfoSwitch [IN] switch whether get time info from model
 * @param  aicoreMetrics [IN]  aicore metrics
 * @param  fd [IN]             pointer to write pipe
 *
 * @retval the aclprofSubscribeConfig pointer
 *
 * @see aclprofDestroySubscribeConfig
 */
ACL_FUNC_VISIBILITY aclprofSubscribeConfig *aclprofCreateSubscribeConfig(int8_t timeInfoSwitch,
    aclprofAicoreMetrics aicoreMetrics, void *fd);

/**
 * @ingroup AscendCL
 * @brief destroy subscribe config
 *
 * @param  profSubscribeConfig [IN]  subscribe config
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclprofCreateSubscribeConfig
 */
ACL_FUNC_VISIBILITY aclError aclprofDestroySubscribeConfig(const aclprofSubscribeConfig *profSubscribeConfig);

/**
 * @ingroup AscendCL
 * @brief create subscribe config
 *
 * @param  opDescSize [OUT]  size of op desc
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclprofGetOpDescSize(size_t *opDescSize);

/**
 * @ingroup AscendCL
 * @brief get op number from subscription data
 *
 * @param  opInfo [IN]     pointer to subscription data
 * @param  opInfoLen [IN]  memory size of subscription data
 * @param  opNumber [OUT]  op number of subscription data
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclprofGetOpNum(const void *opInfo, size_t opInfoLen, uint32_t *opNumber);

/**
 * @ingroup AscendCL
 * @brief get op type from subscription data
 *
 * @param  opInfo [IN]      pointer to subscription data
 * @param  opInfoLen [IN]   memory size of subscription data
 * @param  index [IN]       index of op array in opInfo
 * @param  opType [OUT]     obtained op type string
 * @param  opTypeLen [IN]   obtained length of op type string
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclprofGetOpType(const void *opInfo, size_t opInfoLen, uint32_t index,
    char *opType, size_t opTypeLen);

/**
 * @ingroup AscendCL
 * @brief get op type from subscription data
 *
 * @param  opInfo [IN]      pointer to subscription data
 * @param  opInfoLen [IN]   memory size of subscription data
 * @param  index [IN]       index of op array in opInfo
 * @param  opName [OUT]     obtained op name string
 * @param  opNameLen [IN]   obtained length of op name string
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclprofGetOpName(const void *opInfo, size_t opInfoLen, uint32_t index,
    char *opName, size_t opNameLen);

/**
 * @ingroup AscendCL
 * @brief get start time of specified op from subscription data
 *
 * @param  opInfo [IN]     pointer to subscription data
 * @param  opInfoLen [IN]  memory size of subscription data
 * @param  index [IN]      index of op array in opInfo
 *
 * @retval start time(us) of specified op with timestamp
 * @retval 0 for failed
 */
ACL_FUNC_VISIBILITY uint64_t aclprofGetOpStart(const void *opInfo, size_t opInfoLen, uint32_t index);

/**
 * @ingroup AscendCL
 * @brief get end time of specified op from subscription data
 *
 * @param  opInfo [IN]     pointer to subscription data
 * @param  opInfoLen [IN]  memory size of subscription data
 * @param  index [IN]      index of op array in opInfo
 *
 * @retval end time(us) of specified op with timestamp
 * @retval 0 for failed
 */
ACL_FUNC_VISIBILITY uint64_t aclprofGetOpEnd(const void *opInfo, size_t opInfoLen, uint32_t index);

/**
 * @ingroup AscendCL
 * @brief get excution time of specified op from subscription data
 *
 * @param  opInfo [IN]     pointer to subscription data
 * @param  opInfoLen [IN]  memory size of subscription data
 * @param  index [IN]      index of op array in opInfo
 *
 * @retval execution time(us) of specified op with timestamp
 * @retval 0 for failed
 */
ACL_FUNC_VISIBILITY uint64_t aclprofGetOpDuration(const void *opInfo, size_t opInfoLen, uint32_t index);

/**
 * @ingroup AscendCL
 * @brief get model id from subscription data
 *
 * @param  opInfo [IN]     pointer to subscription data
 * @param  opInfoLen [IN]  memory size of subscription data
 *
 * @retval model id of subscription data
 * @retval 0 for failed
 */
ACL_FUNC_VISIBILITY size_t aclprofGetModelId(const void *opInfo, size_t opInfoLen, uint32_t index);

#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_ACL_PROF_H_
