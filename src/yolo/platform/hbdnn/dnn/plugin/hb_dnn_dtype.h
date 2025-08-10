// Copyright (c) 2021 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef DNN_PLUGIN_HB_DNN_DTYPE_H_
#define DNN_PLUGIN_HB_DNN_DTYPE_H_

#include <cstdint>
#include <cstdlib>

#define HB_DNN_SIZEOF_TYPE(type) \
  (hobot::dnn::TypeSize[static_cast<size_t>(type)])

namespace hobot {
namespace dnn {

enum class TypeFlag : uint32_t {
  kBool = 0U,
  kUInt8 = 1U,
  kInt8 = 2U,
  kUInt16 = 3U,
  kInt16 = 4U,
  kUInt32 = 5U,
  kInt32 = 6U,
  kUInt64 = 7U,
  kInt64 = 8U,
  kFloat16 = 9U,
  kFloat32 = 10U,
  kFloat64 = 11U,
  kUnused
};  // enum TypeFlag

extern size_t TypeSize[static_cast<uint32_t>(TypeFlag::kUnused) + 1];

template <typename DType>
struct DataType {
  static inline TypeFlag kFlag() { return TypeFlag::kUnused; }
};

template <>
struct DataType<bool> {
  static inline TypeFlag kFlag() { return TypeFlag::kBool; }
};

template <>
struct DataType<uint8_t> {
  static inline TypeFlag kFlag() { return TypeFlag::kUInt8; }
};

template <>
struct DataType<int8_t> {
  static inline TypeFlag kFlag() { return TypeFlag::kInt8; }
};

template <>
struct DataType<uint16_t> {
  static inline TypeFlag kFlag() { return TypeFlag::kUInt16; }
};

template <>
struct DataType<int16_t> {
  static inline TypeFlag kFlag() { return TypeFlag::kInt16; }
};

template <>
struct DataType<uint32_t> {
  static inline TypeFlag kFlag() { return TypeFlag::kUInt32; }
};

template <>
struct DataType<int32_t> {
  static inline TypeFlag kFlag() { return TypeFlag::kInt32; }
};

template <>
struct DataType<int64_t> {
  static inline TypeFlag kFlag() { return TypeFlag::kInt64; }
};

template <>
struct DataType<uint64_t> {
  static inline TypeFlag kFlag() { return TypeFlag::kUInt64; }
};

template <>
struct DataType<float> {
  static inline TypeFlag kFlag() { return TypeFlag::kFloat32; }
};

template <>
struct DataType<double> {
  static inline TypeFlag kFlag() { return TypeFlag::kFloat64; }
};

}  // namespace dnn
}  // namespace hobot
#endif  // DNN_PLUGIN_HB_DNN_DTYPE_H_
