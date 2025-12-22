#include <onnxruntime_cxx_api.h>
#include <iostream>

void PrintModelIO(Ort::Session& session)
{
    Ort::AllocatorWithDefaultOptions allocator;

    // -------- Inputs --------
    size_t num_inputs = session.GetInputCount();
    std::cout << "Model Inputs: " << num_inputs << "\n";

    for (size_t i = 0; i < num_inputs; ++i)
    {
        auto input_name = session.GetInputNameAllocated(i, allocator);
        std::cout << input_name.get() << std::endl;

        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType elem_type = tensor_info.GetElementType();
        std::vector<int64_t> shape = tensor_info.GetShape();

        std::cout << "  Element type: " << elem_type << "\n";
        std::cout << "  Shape: [ ";
        for (auto d : shape)
            std::cout << d << " ";
        std::cout << "]\n";
    }

    // -------- Outputs --------
    size_t num_outputs = session.GetOutputCount();
    std::cout << "Model Outputs: " << num_outputs << "\n";

    for (size_t i = 0; i < num_outputs; ++i)
    {
        auto output_name = session.GetOutputNameAllocated(i, allocator);
        std::cout << output_name.get() << std::endl;

        Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType elem_type = tensor_info.GetElementType();
        std::vector<int64_t> shape = tensor_info.GetShape();

        std::cout << "  Element type: " << elem_type << "\n";
        std::cout << "  Shape: [ ";
        for (auto d : shape)
            std::cout << d << " ";
        std::cout << "]\n";
    }
}

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
        PrintModelIO(session);

    }
    catch (const Ort::Exception& e) {
        std::cerr << "ORT ERROR: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
