#include <onnxruntime_cxx_api.h>
#include <iostream>

int main()
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    OrtCUDAProviderOptions cuda_options;
    session_options.AppendExecutionProvider_CUDA(cuda_options);

    const wchar_t* model_path = L"scrfd_model.onnx";

    try {
        Ort::Session session(env, model_path, session_options);
        std::cout << "ONNX Runtime loaded model successfully!" << std::endl;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ORT ERROR: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
