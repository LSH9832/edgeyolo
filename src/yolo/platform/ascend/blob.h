/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2012-2021. All rights reserved.
 * Description: header for blob
*/

#ifndef BLOB_H
#define BLOB_H

#define ENABLE_DVPP_INTERFACE
#include <string>
#include <vector>
#include <iostream>

#include "./acl/ops/acl_dvpp.h"
#include "./acl/acl_base.h"

class BlobDesc {
public:
    BlobDesc(): size_(0)
    {
        shape_.assign(0, 0);
    }

    virtual ~BlobDesc()
    {
        shape_.clear();
        size_ = 0;
    }

    inline uint32_t Size() const
    {
        return size_;
    }

    inline uint32_t Dim(const int i) const
    {
        return shape_[i];
    }

    inline std::string Shape() const
    {
        std::string shapeStr = "";
        for (int i = 0; i < shape_.size(); i++) {
            shapeStr += std::to_string(shape_[i]) + "  ";
        }
        return shapeStr;
    }

    inline uint32_t Offset(const std::vector<uint32_t> dim) const
    {
        uint32_t offset = 0;
        for (int i = 0; i < shape_.size(); i++) {
            offset *= shape_[i];
            if (dim.size() > i) {
                offset += dim[i];
            }
        }

        return offset;
    }

    virtual int Reshape(const std::vector<uint32_t> &dim) = 0;

    virtual int ReshapeLike(const BlobDesc &blobDesc) = 0;

    uint32_t size_;
    std::vector<uint32_t> shape_;
};

template <class T>
class Blob : public BlobDesc {
public:
    Blob() : data_(nullptr) {}

    ~Blob()
    {
        if (data_ != nullptr) {
            (void)acldvppFree(data_);
        }
        if (host_data_ != nullptr) {
            (void)aclrtFree(host_data_);
        }
        data_ = nullptr;
        host_data_ = nullptr;
    }

    T *Data()
    {
        return data_;
    }

    T *HostData() {
        // std::cout << "device2host" << std::endl;
        aclrtMemcpy(host_data_, size_, data_, size_, ACL_MEMCPY_DEVICE_TO_HOST);
        // std::cout << "device2host end" << std::endl;
        return host_data_;
    }

    aclError dataHost2Device() {
        // std::cout << "host2device" << std::endl;
        aclError err = aclrtMemcpy(data_, size_, host_data_, size_, ACL_MEMCPY_HOST_TO_DEVICE);
        // std::cout << "host2device end" << std::endl;
        return err;
    }

    virtual int Reshape(const std::vector<uint32_t> &dim)
    {
        shape_.clear();
        size_ = sizeof(T);
        for (int i = 0; i < dim.size(); i++) {
            shape_.push_back(dim[i]);
            size_ *= dim[i];
        }

        if (Alloc(size_) != 0) {
            ACL_APP_LOG(ACL_ERROR, "Alloc blob data failed!");
            return 1;
        }

        return 0;
    }

    virtual int ReshapeLike(const BlobDesc &blob)
    {
        return Reshape(blob.shape_);
    }

    int Alloc(const uint32_t size)
    {
        size_ = size;
        if (data_ != nullptr) {
            (void)acldvppFree(data_);
            data_ = nullptr;
            ACL_APP_LOG(ACL_WARNING, "blob has been initialized, reinit!");
        }

        if (host_data_ != nullptr) {
            (void)aclrtFree(host_data_);
            ACL_APP_LOG(ACL_WARNING, "host blob has been initialized, reinit!");
        }

        aclError ret = acldvppMalloc((void **)&data_, size);
        if (ret != ACL_ERROR_NONE || data_ == nullptr) {
            ACL_APP_LOG(ACL_ERROR, "aclMalloc fail, size is %zu!", size);
            return 1;
        }

        ret = aclrtMallocHost((void **)&host_data_, size);
        if (ret != ACL_ERROR_NONE || host_data_ == nullptr) {
            ACL_APP_LOG(ACL_ERROR, "aclMalloc fail, size is %zu!", size);
            return 1;
        }

        return 0;
    }

private:
    T * data_;
    T * host_data_;
};

#endif
