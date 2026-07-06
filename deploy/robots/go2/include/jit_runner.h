#pragma once

#include <memory>
#include <string>

#include "isaaclab/algorithms/algorithms.h"

namespace isaaclab
{

std::unique_ptr<Algorithms> make_jit_runner(const std::string& model_path);

} // namespace isaaclab
