#include "ONNXModel.h"

ONNXModel::ONNXModel(Ort::Env& env, const ORTCHAR_T* path, std::vector<std::int64_t> input_shape): input_shape_(std::move(input_shape)) {
	size_t input_tensor_size = input_shape_[1] * input_shape_[2] * input_shape_[3];
	input_buffer.resize(input_tensor_size);

	options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	session = Ort::Session(env, path, options);
	mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);


	Ort::AllocatorWithDefaultOptions allocator;
	
	input_name_owner = session.GetInputNameAllocated(0, allocator);
	input_names = { input_name_owner.get() };

	for (size_t i = 0; i < session.GetOutputCount(); ++i) {
		output_name_owners.push_back(session.GetOutputNameAllocated(i, allocator));
		output_names.push_back(output_name_owners.back().get());
	}
}

float* ONNXModel::input_data() { return input_buffer.data(); }

std::vector<Ort::Value> ONNXModel::run() {
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
		mem_info,
		input_buffer.data(), input_buffer.size(),
		input_shape_.data(), input_shape_.size()
	);

	return session.Run(
		Ort::RunOptions{nullptr},input_names.data(),
		&input_tensor, 1, output_names.data(), output_names.size()
	);
}