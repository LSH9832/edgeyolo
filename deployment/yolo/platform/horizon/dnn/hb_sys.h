// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef DNN_HB_SYS_H_
#define DNN_HB_SYS_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum {
  HB_BPU_CORE_ANY = 0,
  HB_BPU_CORE_0 = (1 << 0),
  HB_BPU_CORE_1 = (1 << 1)
} hbBPUCore;

typedef enum {
  HB_DSP_CORE_ANY = 0,
  HB_DSP_CORE_0 = (1 << 0),
  HB_DSP_CORE_1 = (1 << 1)
} hbDSPCore;

typedef struct {
  uint64_t phyAddr;
  void *virAddr;
  uint32_t memSize;
} hbSysMem;

typedef enum {
  HB_SYS_MEM_CACHE_INVALIDATE = 1,
  HB_SYS_MEM_CACHE_CLEAN = 2
} hbSysMemFlushFlag;

typedef enum {
  HB_SYS_SUCCESS = 0,
  HB_SYS_INVALID_ARGUMENT = -6000129,
  HB_SYS_OUT_OF_MEMORY = -6000130,
  HB_SYS_REGISTER_MEM_FAILED = -6000131,
} hbSysStatus;

/**
 * Allocate system memory
 * @param[out] mem
 * @param[in] size
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbSysAllocMem(hbSysMem *mem, uint32_t size);

/**
 * Allocate cachable system memory
 * @param[out] mem
 * @param[in] size
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbSysAllocCachedMem(hbSysMem *mem, uint32_t size);

/**
 * Flush cachable system memory
 * @param[in] mem
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbSysFlushMem(hbSysMem *mem, int32_t flag);

/**
 * Write mem
 * @param[out] mem
 * @param[in] mem
 * @param[in] size
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbSysWriteMem(hbSysMem *dest, char *src, uint32_t size);

/**
 * Read mem
 * @param[out] mem
 * @param[in] mem
 * @param[in] size
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbSysReadMem(char *dest, hbSysMem *src, uint32_t size);

/**
 * Free mem
 * @param[in] mem
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbSysFreeMem(hbSysMem *mem);

/**
 * Register mem (vio etc) so that it can be used by BPU
 * @deprecated kept for compatibility purpose
 * @param[in] mem
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbSysRegisterMem(hbSysMem *mem);

/**
 * Unregister mem (already registered vio mem etc)
 * @deprecated kept for compatibility purpose
 * @param[in] mem
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbSysUnregisterMem(hbSysMem *mem);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // DNN_HB_SYS_H_
