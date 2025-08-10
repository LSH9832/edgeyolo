// Copyright (c) 2021 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef DNN_PLUGIN_HB_DNN_NDARRAY_H_
#define DNN_PLUGIN_HB_DNN_NDARRAY_H_

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "./hb_dnn_dtype.h"
#include "./hb_dnn_tuple.h"

namespace hobot {
namespace dnn {

class Chunk;

class NDArray {
 public:
  NDArray() = default;

  explicit NDArray(const TShape &shape, TypeFlag dtype = TypeFlag::kFloat32);

  explicit NDArray(void *dptr, const TShape &shape,
                   TypeFlag dtype = TypeFlag::kFloat32);

  NDArray(const NDArray &) = default;

  NDArray &operator=(const NDArray &) = default;

  virtual ~NDArray() = default;

  /**
   * @return pointer of memory in ndarray.
   */
  template <typename DType>
  DType *Dptr() const;

  /**
   * @return raw pointer of memory
   */
  void *RawData() const;

  /**
   * @return raw pointer of hbSysMem memory
   */
  void *VirAddr() const;

  /**
   * @return physical address of hbSysMem memory
   */
  uint64_t PhyAddr() const;

  /**
   * Deep copy of the ndarray
   * @return a new NDArray instance
   */
  NDArray DeepCopy() const;

  /**
   * Copy to the ndarray
   * @param[in] data source data buffer
   * @param[in] size data size to copy
   * @return successful or not
   */
  bool SyncCopyFrom(const void *data, size_t size);

  /**
   * Copy from the ndarray
   * @param[in] data destination data buffer
   * @param[in] size data size to copy
   * return successful or not
   */
  bool SyncCopyTo(void *data, size_t size) const;

  /**
   * Save ndarray in binary or plain text
   * @param[in] os ostream
   * @param[in] is_txt save as plain text or binary
   * @return successful or not
   */
  bool Save(std::ostream *os, bool is_txt = false) const;

  /**
   * Reset ptr of ndarray
   * @param[in] dptr input pointer
   */
  void ResetPtr(void *dptr);

  /**
   * @return shape of the ndarray
   */
  inline const TShape &Shape() const { return shape_; }

  /**
   * @return number of dimension of the ndarray
   */
  inline uint32_t NDim() const { return shape_.NDim(); }

  /**
   * @return number of elements in the ndarray.
   */
  inline uint32_t Size() const { return shape_.ProdSize(); }

  /**
   * Get canonical axis index
   * @param[in] axis: axis should be in the range [-NDim(), NDim())
   * @return canonical axis index
   */
  int32_t CanonicalAxis(int32_t axis) const;

  /**
   * @return data type of the ndarray
   */
  inline TypeFlag Dtype() const { return dtype_; }

  /**
   * @return if ndarray is null or not
   */
  inline bool IsNone() const { return ptr_ == nullptr; }

  /**
   * Slice the first dimension of the array const
   * @param[in] idx slice index of dim 0, it should be in the range [0, Shape()[0]-1)
   * @return sliced ndarray
   */
  NDArray operator[](uint32_t idx) const;

  /**
   * Create a new type of ndarray from current ndarray
   * @param[in] shape
   * @param[in] dtype
   * @return new ndarray
   */
  NDArray AsArray(const TShape &shape, TypeFlag dtype) const;

  /**
   * Reshape ndarray
   * @param[in] shape
   * @return a ndarray with given shape if success a none ndarray otherwise
   */
  NDArray Reshape(const TShape &shape) const;

  /**
   * Flatten the all dimensions, return a 1D ndarray
   * @return 1D ndarray
   */
  NDArray Flatten() const;

 private:
  std::shared_ptr<Chunk> ptr_;

  TShape shape_;

  TypeFlag dtype_{TypeFlag::kUnused};

  size_t offset_{0U};
};  // class NDArray
}  // namespace dnn
}  // namespace hobot

#endif  // DNN_PLUGIN_HB_DNN_NDARRAY_H_
