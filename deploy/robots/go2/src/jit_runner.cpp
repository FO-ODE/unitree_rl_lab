#include "jit_runner.h"

#include <torch/script.h>
#include <torch/torch.h>

namespace isaaclab
{

class JitRunner final : public Algorithms
{
public:
    explicit JitRunner(const std::string& model_path)
    {
        if (!torch::cuda::is_available()) {
            throw std::runtime_error("CUDA is not available for TorchScript policy inference.");
        }
        module = torch::jit::load(model_path);
        module.to(device);
        module.eval();
        action.resize(12, 0.0f);
    }

    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) override
    {
        for (const auto& name : input_names) {
            if (obs.find(name) == obs.end()) {
                throw std::runtime_error("Input name " + name + " not found in observations.");
            }
        }

        std::vector<torch::jit::IValue> inputs;
        for (const auto& name : input_names) {
            auto& input_data = obs.at(name);
            auto tensor = torch::from_blob(input_data.data(), {1, static_cast<long>(input_data.size())}, torch::kFloat32)
                .clone()
                .to(device);
            inputs.push_back(tensor);
        }

        torch::NoGradGuard no_grad;
        auto output = module.forward(inputs).toTensor().contiguous().view({-1}).cpu();
        auto output_accessor = output.accessor<float, 1>();

        std::lock_guard<std::mutex> lock(act_mtx_);
        action.resize(output.size(0));
        for (int64_t i = 0; i < output.size(0); ++i) {
            action[i] = output_accessor[i];
        }
        return action;
    }

private:
    torch::Device device = torch::Device(torch::kCUDA);
    torch::jit::script::Module module;
    std::vector<std::string> input_names = {
        "policy_raw_obs",
        "policy_history_obs",
        "policy_terrain_obs",
    };
};

std::unique_ptr<Algorithms> make_jit_runner(const std::string& model_path)
{
    return std::make_unique<JitRunner>(model_path);
}

} // namespace isaaclab
