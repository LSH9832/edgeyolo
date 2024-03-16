/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2012-2021. All rights reserved.
 * Description: source for davinci net
*/

#include <detect/mdc/davinci_net.h>

DavinciNet::DavinciNet()
    : modelId_(0), modelMemPtr_(nullptr), modelWeightPtr_(nullptr),
      modelDesc_(nullptr), input_(nullptr), output_(nullptr), deviceId_(0),
      context_(nullptr), stream_(nullptr),  isLoaded_(false)
{
    blobMap.clear();
}

DavinciNet::~DavinciNet()
{
    UnloadModel();
    DestroyDesc();
    DestroyInputTensor();
    DestroyOutputTensor();
    DestroyContextAndStream();
}

int DavinciNet::Init(const std::string &modelPath, const int32_t deviceId)
{
    deviceId_ = deviceId;

    int ret = CreateContextAndStream();
    if (ret != 0) {
        ACL_APP_LOG(ACL_ERROR, "execute CreateContextAndStream failed");
        return 1;
    }

    ret = LoadModelFromFile(modelPath);
    if (ret != 0) {
        ACL_APP_LOG(ACL_ERROR, "execute LoadModelFromFile failed");
        return 1;
    }

    ret = CreateDesc();
    if (ret != 0) {
        ACL_APP_LOG(ACL_ERROR, "execute CreateDesc failed");
        return 1;
    }

    ret = CreateInputTensor();
    if (ret != 0) {
        ACL_APP_LOG(ACL_ERROR, "execute CreateInput failed");
        return 1;
    }

    ret = CreateOutputTensor();
    if (ret != 0) {
        ACL_APP_LOG(ACL_ERROR, "execute CreateOutput failed");
        return 1;
    }

    return 0;
}

int DavinciNet::CreateContextAndStream()
{
    // create context (set current)
    aclError ret = aclrtCreateContext(&context_, deviceId_);
    if (ret != ACL_ERROR_NONE) {
        ACL_APP_LOG(ACL_ERROR, "acl create context failed");
        return 1;
    }
    ACL_APP_LOG(ACL_INFO, "create context success");

    // create stream
    ret = aclrtCreateStream(&stream_);
    if (ret != ACL_ERROR_NONE) {
        ACL_APP_LOG(ACL_ERROR, "acl create stream failed");
        return 1;
    }
    ACL_APP_LOG(ACL_INFO, "create stream success");

    // get run mode
    aclrtRunMode runMode;
    ret = aclrtGetRunMode(&runMode);
    if (ret != ACL_ERROR_NONE) {
        ACL_APP_LOG(ACL_ERROR, "acl get run mode failed");
        return 1;
    }

    return 0;
}

int DavinciNet::LoadModelFromFile(const std::string &modelPath)
{
    if (isLoaded_) {
        ACL_APP_LOG(ACL_ERROR, "has already loaded a model");
        return 1;
    }

    size_t modelMemSize;
    size_t modelWeightSize;

    aclError ret = aclmdlQuerySize(modelPath.c_str(), &modelMemSize, &modelWeightSize);
    if (ret != ACL_ERROR_NONE) {
        ACL_APP_LOG(ACL_ERROR, "query model failed, model file is %s", modelPath);
        return 1;
    }

    ret = aclrtMalloc(&modelMemPtr_, modelMemSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
        ACL_APP_LOG(ACL_ERROR, "malloc buffer for mem failed, require size is %zu", modelMemSize);
        return 1;
    }

    ret = aclrtMalloc(&modelWeightPtr_, modelWeightSize,
                      ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
        ACL_APP_LOG(ACL_ERROR, "malloc buffer for weight failed, require size is %zu", modelWeightSize);
        return 1;
    }

    ret = aclmdlLoadFromFileWithMem(modelPath.c_str(), &modelId_, modelMemPtr_,
                                    modelMemSize, modelWeightPtr_,
                                    modelWeightSize);
    if (ret != ACL_ERROR_NONE) {
        ACL_APP_LOG(ACL_ERROR, "load model from file failed, model file is %s", modelPath);
        return 1;
    }

    isLoaded_ = true;
    ACL_APP_LOG(ACL_INFO, "load model %s success", modelPath);
    return 0;
}

int DavinciNet::CreateDesc()
{
    modelDesc_ = aclmdlCreateDesc();
    if (modelDesc_ == nullptr) {
        ACL_APP_LOG(ACL_ERROR, "create model description failed");
        return 1;
    }

    aclError ret = aclmdlGetDesc(modelDesc_, modelId_);
    if (ret != ACL_ERROR_NONE) {
        ACL_APP_LOG(ACL_ERROR, "get model description failed");
        return 1;
    }

    ACL_APP_LOG(ACL_INFO, "create model description success");

    return 0;
}

void DavinciNet::DisplayTensorInfo(const aclmdlIODims &dims, const aclDataType dtype, const bool input=true)
{
    // print tensor message
    std::cout << "    name: " << dims.name << std::endl
              << "    data type: " << static_cast<int>(dtype) << std::endl
              << "    shape info: ";
    std::vector<int> this_dim;
    for (size_t i = 0; i < dims.dimCount; i++) {
        std::cout << "dim[" << i << "]: " << dims.dims[i] << "\t";
        this_dim.push_back(dims.dims[i]);
    }
    if (input) inputs_dims.push_back(this_dim);
    else outputs_dims.push_back(this_dim);
    std::cout << std::endl;
}

int DavinciNet::CreateInputTensor()
{
    if (modelDesc_ == nullptr) {
        ACL_APP_LOG(ACL_ERROR, "no model description, create input failed");
        return 1;
    }

    input_ = aclmdlCreateDataset();
    if (input_ == nullptr) {
        ACL_APP_LOG(ACL_ERROR, "can't create dateset, create input failed");
        return 1;
    }

    std::cout << "input tensor info:" << std::endl;
    size_t inputSize = aclmdlGetNumInputs(modelDesc_);
    input_names.clear();
    for (size_t i = 0; i < inputSize; ++i) {
        std::cout << "input[" << i << "]:" << std::endl;

        // get input tensor dims
        aclmdlIODims dims;
        aclError ret = aclmdlGetInputDims(modelDesc_, i, &dims);
        if (ret != ACL_ERROR_NONE) {
            ACL_APP_LOG(ACL_ERROR, "can't get %luth input dims", i);
            return 1;
        }

        // get output date type
        aclDataType dtype = aclmdlGetInputDataType(modelDesc_, i);
        DisplayTensorInfo(dims, dtype);
        input_names.push_back(dims.name);

        // Create tensor
        std::shared_ptr<Blob<void>> inputBlob = std::make_shared<Blob<void>>();
        size_t buffSize = aclmdlGetInputSizeByIndex(modelDesc_, i);
        std::cout << "input buffSize: " << buffSize << std::endl;

        ret = inputBlob->Alloc(buffSize);
        if (ret != 0) {
            ACL_APP_LOG(ACL_ERROR, "Alloc input blob failed!");
            return 1;
        }
        blobMap[dims.name] = inputBlob;

        aclDataBuffer *inputData = aclCreateDataBuffer(inputBlob->Data(),
                                                       inputBlob->Size());
        if (ret != ACL_ERROR_NONE) {
            ACL_APP_LOG(ACL_ERROR, "can't create data buffer, create input failed");
            return 1;
        }

        ret = aclmdlAddDatasetBuffer(input_, inputData);
        if (ret != ACL_ERROR_NONE) {
            ACL_APP_LOG(ACL_ERROR, "can't add data buffer, create input failed");
            aclDestroyDataBuffer(inputData);
            return 1;
        }
    }

    return 0;
}

int DavinciNet::CreateOutputTensor()
{
    if (modelDesc_ == nullptr) {
        ACL_APP_LOG(ACL_ERROR, "no model description, create ouput failed");
        return 1;
    }

    output_ = aclmdlCreateDataset();
    if (output_ == nullptr) {
        ACL_APP_LOG(ACL_ERROR, "can't create dataset, create output failed");
        return 1;
    }

    std::cout << "output tensor info:" << std::endl;
    size_t outputSize = aclmdlGetNumOutputs(modelDesc_);
    output_names.clear();
    for (size_t i = 0; i < outputSize; ++i) {
        std::cout << "output[" << i << "]:" << std::endl;
        // get output tensor dims
        aclmdlIODims dims;
        aclError ret = aclmdlGetOutputDims(modelDesc_, i, &dims);
        if (ret != ACL_ERROR_NONE) {
            ACL_APP_LOG(ACL_ERROR, "can't get %luth output dims", i);
            return 1;
        }

        // get output date type
        aclDataType dtype = aclmdlGetOutputDataType(modelDesc_, i);
        DisplayTensorInfo(dims, dtype, false);
        output_names.push_back(dims.name);

        // Create tensor
        std::shared_ptr<Blob<void>> outputBlob = std::make_shared<Blob<void>>();
        size_t buffSize = aclmdlGetOutputSizeByIndex(modelDesc_, i);

        std::cout << "output buffSize: " << buffSize << std::endl;
        ret = outputBlob->Alloc(buffSize);
        if (ret != 0) {
            ACL_APP_LOG(ACL_ERROR, "Alloc output blob failed!");
            return 1;
        }
        blobMap[dims.name] = outputBlob;

        aclDataBuffer* outputData = aclCreateDataBuffer(outputBlob->Data(),
                                                        outputBlob->Size());
        if (ret != ACL_ERROR_NONE) {
            ACL_APP_LOG(ACL_ERROR, "can't create data buffer, create output failed");
            return 1;
        }

        ret = aclmdlAddDatasetBuffer(output_, outputData);
        if (ret != ACL_ERROR_NONE) {
            ACL_APP_LOG(ACL_ERROR, "can't add data buffer, create output failed");
            aclDestroyDataBuffer(outputData);
            return 1;
        }
    }

    return 0;
}

void DavinciNet::DestroyDesc()
{
    if (modelDesc_ != nullptr) {
        (void)aclmdlDestroyDesc(modelDesc_);
        modelDesc_ = nullptr;
    }
}

void DavinciNet::DestroyInputTensor()
{
    if (input_ == nullptr) {
        return;
    }
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input_); ++i) {
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(input_, i);
        aclDestroyDataBuffer(dataBuffer);
    }

    (void)aclmdlDestroyDataset(input_);
    input_ = nullptr;
}

void DavinciNet::DestroyOutputTensor()
{
    if (output_ == nullptr) {
        return;
    }

    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, i);
        (void)aclDestroyDataBuffer(dataBuffer);
    }

    (void)aclmdlDestroyDataset(output_);
    output_ = nullptr;
}

void DavinciNet::UnloadModel()
{
    if (!isLoaded_) {
        ACL_APP_LOG(ACL_WARNING, "no model had been loaded, no need to unload");
        return;
    }

    aclError ret = aclmdlUnload(modelId_);
    if (ret != ACL_ERROR_NONE) {
        ACL_APP_LOG(ACL_ERROR, "unload model failed, modelId is %u", modelId_);
    }

    if (modelDesc_ != nullptr) {
        (void)aclmdlDestroyDesc(modelDesc_);
        modelDesc_ = nullptr;
    }

    if (modelMemPtr_ != nullptr) {
        aclrtFree(modelMemPtr_);
        modelMemPtr_ = nullptr;
    }

    if (modelWeightPtr_ != nullptr) {
        aclrtFree(modelWeightPtr_);
        modelWeightPtr_ = nullptr;
    }

    isLoaded_ = false;
    ACL_APP_LOG(ACL_INFO, "unload model success, modelId is %u", modelId_);
}

int DavinciNet::Inference()
{
    aclError ret = aclrtSetCurrentContext(context_);
    if (ret != ACL_ERROR_NONE) {
        ACL_APP_LOG(ACL_ERROR, "acl set current context failed!");
        return 1;
    }

    ret = aclmdlExecute(modelId_, input_, output_);
    if (ret != ACL_ERROR_NONE) {
        ACL_APP_LOG(ACL_ERROR, "execute model failed, modelId is %u", modelId_);
        return 1;
    }

    return 0;
}

std::shared_ptr<BlobDesc> DavinciNet::GetBlob(const std::string &blobName)
{
    auto it = blobMap.find(blobName);
    if (it != blobMap.end()) {
        return it->second;
    }
    return nullptr;
}

void DavinciNet::DestroyContextAndStream()
{
    aclError ret;
    if (stream_ != nullptr) {
        ret = aclrtDestroyStream(stream_);
        if (ret != ACL_ERROR_NONE) {
            ACL_APP_LOG(ACL_ERROR, "destroy stream failed");
        }
        stream_ = nullptr;
    }
    ACL_APP_LOG(ACL_INFO, "End to destroy stream");

    if (context_ != nullptr) {
        ret = aclrtDestroyContext(context_);
        if (ret != ACL_ERROR_NONE) {
            ACL_APP_LOG(ACL_ERROR, "destroy context failed");
        }
        context_ = nullptr;
    }
    ACL_APP_LOG(ACL_INFO, "End to destroy context");
}
