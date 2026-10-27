#pragma once
#include <daxa/utils/task_graph.hpp>

namespace sintergpu {
// Caller registers task attachment accesses and supplies reflected push bytes.
inline void record_compute(daxa::TaskInterface const &task,
                           daxa::ComputePipeline const &pipeline,
                           daxa::DispatchInfo const &groups,
                           daxa::PushConstantInfo const &push = {}) {
    task.recorder.set_pipeline(pipeline);
    if (push.size) task.recorder.push_constant_vptr(push);
    task.recorder.dispatch(groups);
}
} // namespace sintergpu
