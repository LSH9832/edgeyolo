// Copyright (c) 2021 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef DNN_PLUGIN_HB_DNN_TUPLE_H_
#define DNN_PLUGIN_HB_DNN_TUPLE_H_

#include <algorithm>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace hobot {
namespace dnn {

template <typename ValueType>
class Tuple {
 public:
  Tuple() = default;

  inline virtual ~Tuple() { delete[] heap_data_; }

  /**
   * DeepCopy constructor from another tuple
   * @param[in] s the source tuple
   */
  inline Tuple(const Tuple<ValueType> &s) { this->Assign(s.Begin(), s.End()); }

  /**
   * Constructor from initializer list
   * @param[in] init the initializer_
   */
  inline Tuple(std::initializer_list<ValueType> init) {
    this->Assign(init.begin(), init.end());
  }

  /**
   * Move constructor from Tuple
   * @param[in] src the source shape
   */
  inline Tuple(Tuple<ValueType> &&src) noexcept { this->Swap(src); }

  /**
   * Construct the Tuple from content of iterator
   * @param[in] begin the beginning of iterator
   * @param[in] end end the end of the iterator
   */
  template <typename RandomAccessIterator>
  inline Tuple(RandomAccessIterator begin, RandomAccessIterator end) {
    this->Assign(begin, end);
  }

  /**
   * Assign content to tuple from iterator.
   * @param[in] begin the beginning of iterator
   * @param[in] end end the end of the iterator
   */
  template <typename RandomAccessIterator>
  inline void Assign(RandomAccessIterator begin, RandomAccessIterator end) {
    this->Resize(static_cast<uint32_t>(end - begin));
    std::copy(begin, end, this->Begin());
  }

  /**
   * Swap current object with other
   * @param[in] other another object to be swapped.
   */
  inline void Swap(Tuple<ValueType> &other) {
    std::swap(ndim_, other.ndim_);
    std::swap(capacity_, other.capacity_);
    std::swap(heap_data_, other.heap_data_);
    std::swap(stack_data_, other.stack_data_);
  }

  /**
   * Assignment from another tuple.
   * @param[in] src source tuple
   * @return reference of self
   */
  inline Tuple<ValueType> &operator=(const Tuple<ValueType> &src) & {
    this->Assign(src.Begin(), src.End());
    return *this;
  }

  /**
   * Move assignment from rvalue of another tuple.
   * @param[in] src source tuple
   * @return reference of self
   */
  inline Tuple<ValueType> &operator=(Tuple<ValueType> &&src) & {
    Tuple<ValueType>(std::move(src)).Swap(*this);
    return *this;
  }

  /**
   * Assignment from initializer list
   * @param[in] init the source initializer list
   * @return reference of self
   */
  inline Tuple<ValueType> &operator=(std::initializer_list<ValueType> init) & {
    this->Assign(init.begin(), init.end());
    return *this;
  }

  /**
   * @param[in] s the tuple to compare against
   * @return whether two tuple equals
   */
  inline bool operator==(const Tuple<ValueType> &s) const {
    if (ndim_ != s.ndim_) {
      return false;
    }
    return std::equal(Begin(), End(), s.Begin());
  }

  /**
   * @param[in] s the tuple to compare against
   * @return whether two tuple not equal
   */
  inline bool operator!=(const Tuple<ValueType> &s) const {
    return !(*this == s);
  }

  /**
   * @return the begin data pointer to content of the tuple
   */
  inline const ValueType *Begin() const {
    return (ndim_ <= kStackCache) ? stack_data_ : heap_data_;
  }

  /**
   * @return the begin data pointer to content of the tuple
   */
  inline ValueType *Begin() {
    return (ndim_ <= kStackCache) ? stack_data_ : heap_data_;
  }

  /**
   * @return the data pointer to end of the tuple
   */
  inline const ValueType *End() const { return (Begin() + ndim_); }

  /**
   * @return the data pointer to end of the tuple
   */
  inline ValueType *End() { return (Begin() + ndim_); }

  /**
   * @return number of dimension of the tuple
   */
  inline uint32_t NDim() const { return ndim_; }

  /**
   * Get corresponding index
   * @param[in] i dimension index
   * @return the corresponding dimension size
   */
  inline ValueType &operator[](uint32_t i) { return Begin()[i]; }

  /**
   * Get corresponding index
   * @param[in] i dimension index
   * @return the corresponding dimension size
   */
  inline const ValueType &operator[](uint32_t i) const { return Begin()[i]; }

 protected:
  inline void Resize(uint32_t dim) {
    if (dim > capacity_) {
      delete[] heap_data_;
      capacity_ = dim;
      heap_data_ = new ValueType[static_cast<size_t>(capacity_)];
    } else if (dim <= kStackCache) {
      capacity_ = kStackCache;
    } else {
      // do nothing
    }
    ndim_ = dim;
  }

 protected:
  static const uint32_t kStackCache = 4U;

  // number of dimension of the tuple
  uint32_t ndim_{0};

  // tuple capacity
  uint32_t capacity_{kStackCache};

  // in stack space when ndim is small (less equal than 4)
  // to avoid frequent dynamic malloc/free for heap space
  ValueType stack_data_[kStackCache]{};

  ValueType *heap_data_{nullptr};
};  // class Tuple

/**
 * A class that is used to represent shape of tensor.
 */
class TShape : public Tuple<uint32_t> {
 public:
  TShape() = default;

  ~TShape() override = default;

  /**
   * Constructor to construct a shape with all 1.
   * @param[in] ndim the number of dimension
   */
  explicit TShape(uint32_t ndim) : Tuple() {
    this->Resize(ndim);
    std::fill_n(Begin(), ndim, 1);
  }

  /**
   * Constructor from initializer list
   * @param[in] init the initializer_list
   */
  TShape(std::initializer_list<uint32_t> init) : Tuple() {
    this->Assign(init.begin(), init.end());
  }

  /**
   * DeepCopy constructor of TShape
   * @param[in] s source shape
   */
  TShape(const TShape &s) : Tuple() { this->Assign(s.Begin(), s.End()); }

  /**
   * Move constructor
   * @param[in] s source shape
   */
  TShape(TShape &&s) noexcept : Tuple() { this->Swap(s); }

  /**
   * Assignment function from tshape
   * @param[in] src source shape.
   * @return reference of self
   */
  inline TShape &operator=(const TShape &src) & {
    if (this == &src) {
      return *this;
    }
    this->Assign(src.Begin(), src.End());
    return *this;
  }

  /**
   * Move assignment function from tshape
   * @param[in] src source shape.
   * @return reference of self
   */
  inline TShape &operator=(TShape &&src) & noexcept {
    src.Swap(*this);
    return *this;
  }

  /**
   * Construct the tshape from content of iterator
   * @param[in] begin the beginning of iterator
   * @param[in] end end the end of the iterator
   */
  template <typename RandomAccessIterator>
  inline TShape(RandomAccessIterator begin, RandomAccessIterator end)
      : Tuple() {
    this->Assign(begin, end);
  }

  /**
   * @return TShape infomation of string
   */
  std::string const Str() const {
    std::stringstream os;
    os << '(';
    auto const *const begin = Begin();
    auto const *const end = End();
    for (auto const *it = begin; it != end; ++it) {
      if (it != begin) {
        os << ',';
      }
      os << *it;
    }
    // python style tuple
    if (NDim() == 1U) {
      os << ',';
    }
    os << ')';
    return os.str();
  }

  /**
   * @return TShape infomation of string
   */
  std::string Str() {
    std::stringstream os;
    os << '(';
    auto const *const begin = Begin();
    auto const *const end = End();
    for (auto const *it = begin; it != end; ++it) {
      if (it != begin) {
        os << ',';
      }
      os << *it;
    }
    // python style tuple
    if (NDim() == 1U) {
      os << ',';
    }
    os << ')';
    return os.str();
  }

  /**
   * @return product size in [0, ndim_)
   */
  inline uint32_t ProdSize() const { return ProdSize(0U, NDim()); }

  /**
  *
  * @param begin start dimesion
  * @return product size in [begin, ndim_)
  */
  inline uint32_t ProdSize(uint32_t begin) const {
    return ProdSize(begin, NDim());
  }

  /**
   * @param[in] begin start dimension
   * @param[in] end end dimension
   * @return product size in [start, end)
   */
  inline uint32_t ProdSize(uint32_t begin, uint32_t end) const {
    uint32_t size{1U};
    auto *data = Begin();
    for (uint32_t i{begin}; i < end; ++i) {
      size *= data[i];
    }
    return size;
  }

  /**
   * @return the begin data pointer to content of the tuple
   */
  inline const uint32_t *Data() const { return Begin(); }

  /**
   * @return the begin data pointer to content of the tuple
   */
  inline uint32_t *Data() { return Begin(); }

  /**
   * Check if it is equal to input shape
   * @param[in] s
   * @return is equal or not 
   */
  inline bool operator==(const TShape &s) const {
    if (NDim() != s.NDim()) {
      return false;
    }
    return std::equal(Begin(), End(), s.Begin());
  }

  /**
   * Check if it is not equal to input shape
   * @param[in] s
   * @return is equal or not 
   */
  inline bool operator!=(const TShape &s) const { return !(*this == s); }
};  // class TShape

}  // namespace dnn
}  // namespace hobot

#endif  // DNN_PLUGIN_HB_DNN_TUPLE_H_
