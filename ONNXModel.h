#pragma once
#include <onnxruntime_cxx_api.h>

class ONNXModel
{
public:
	ONNXModel(Ort::Env& env, const ORTCHAR_T* path, std::vector<std::int64_t> input_shape);

	float* input_data();

	std::vector<Ort::Value> run();

private:
	Ort::SessionOptions options;
	Ort::Session session{ nullptr };
	Ort::MemoryInfo mem_info{ nullptr };
	Ort::AllocatedStringPtr input_name_owner;
	std::vector<Ort::AllocatedStringPtr> output_name_owners;
	std::vector<const char*> input_names, output_names;
	std::vector<float> input_buffer;
	std::vector<std::int64_t> input_shape_;
};

