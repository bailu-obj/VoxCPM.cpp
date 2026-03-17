#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "voxcpm/backend.h"
#include "voxcpm/context.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

using Catch::Approx;

namespace voxcpm {
namespace test {
namespace {

std::unique_ptr<VoxCPMBackend> try_create_cuda_backend() {
    try {
        return std::make_unique<VoxCPMBackend>(BackendType::CUDA, 2);
    } catch (const std::exception& e) {
        WARN("CUDA backend unavailable, skipping test: " << e.what());
        return nullptr;
    }
}

bool all_finite(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(), [](float value) {
        return std::isfinite(value);
    });
}

std::vector<float> conv_transpose_1d_reference(const std::vector<float>& weight,
                                               const std::vector<float>& input,
                                               int kernel,
                                               int cout,
                                               int cin,
                                               int length,
                                               int stride) {
    const int output_len = (length - 1) * stride + kernel;
    std::vector<float> output(static_cast<size_t>(output_len * cout), 0.0f);

    for (int co = 0; co < cout; ++co) {
        for (int ci = 0; ci < cin; ++ci) {
            const size_t weight_base = static_cast<size_t>(ci * cout + co) * static_cast<size_t>(kernel);
            const size_t input_base = static_cast<size_t>(ci) * static_cast<size_t>(length);
            for (int t = 0; t < length; ++t) {
                const float in = input[input_base + static_cast<size_t>(t)];
                for (int k = 0; k < kernel; ++k) {
                    output[static_cast<size_t>(co * output_len + t * stride + k)] +=
                        weight[weight_base + static_cast<size_t>(k)] * in;
                }
            }
        }
    }

    return output;
}

}  // namespace

TEST_CASE("CUDA backend sums contiguous materialized view copies", "[cuda][sum][backend]") {
    auto backend = try_create_cuda_backend();
    if (!backend) {
        return;
    }

    VoxCPMContext graph_ctx(ContextType::Graph, 128, 1024);
    ggml_tensor* src = graph_ctx.new_tensor_2d(GGML_TYPE_F32, 8, 4);
    ggml_tensor* dst_base = graph_ctx.new_tensor_3d(GGML_TYPE_F32, 8, 8, 1);
    ggml_set_input(src);

    ggml_tensor* dst_view = ggml_view_2d(graph_ctx.raw_context(), dst_base, 8, 4, dst_base->nb[1], 0);
    ggml_tensor* copied = ggml_cpy(graph_ctx.raw_context(), src, dst_view);
    ggml_tensor* reduced = ggml_sum(graph_ctx.raw_context(), ggml_cont(graph_ctx.raw_context(), copied));
    ggml_set_output(reduced);

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, reduced);
    backend->reserve_compute_memory(graph, "test.cuda.sum");
    backend->alloc_graph(graph, "test.cuda.sum");

    std::vector<float> input(32, 0.0f);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i + 1);
    }
    backend->tensor_set(src, input.data(), 0, input.size() * sizeof(float));

    REQUIRE(backend->compute(graph) == GGML_STATUS_SUCCESS);

    float actual = 0.0f;
    backend->tensor_get(reduced, &actual, 0, sizeof(actual));
    REQUIRE(std::isfinite(actual));
    REQUIRE(actual == Approx(528.0f));
}

TEST_CASE("CUDA im2col handles long 1D widths", "[cuda][im2col][backend]") {
    auto backend = try_create_cuda_backend();
    if (!backend) {
        return;
    }

    constexpr int64_t kWidth = 70000;
    constexpr int64_t kKernel = 7;

    VoxCPMContext graph_ctx(ContextType::Graph, 128, 1024);
    ggml_tensor* kernel = graph_ctx.new_tensor_3d(GGML_TYPE_F32, kKernel, 1, 1);
    ggml_tensor* input = graph_ctx.new_tensor_3d(GGML_TYPE_F32, kWidth, 1, 1);
    ggml_set_input(input);

    ggml_tensor* im2col = ggml_im2col(graph_ctx.raw_context(),
                                      kernel,
                                      input,
                                      1,
                                      0,
                                      0,
                                      0,
                                      1,
                                      0,
                                      false,
                                      GGML_TYPE_F32);
    ggml_set_output(im2col);

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, im2col);
    backend->reserve_compute_memory(graph, "test.cuda.im2col");
    backend->alloc_graph(graph, "test.cuda.im2col");

    std::vector<float> input_data(static_cast<size_t>(kWidth), 1.0f);
    backend->tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    REQUIRE(backend->compute(graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(static_cast<size_t>(kKernel) * 2, 0.0f);
    backend->tensor_get(im2col, actual.data(), 0, actual.size() * sizeof(float));
    REQUIRE(all_finite(actual));
    for (float value : actual) {
        REQUIRE(value == Approx(1.0f));
    }
}

TEST_CASE("CUDA conv_transpose_1d matches reference for F16 weights", "[cuda][conv_transpose_1d][backend]") {
    auto backend = try_create_cuda_backend();
    if (!backend) {
        return;
    }

    constexpr int kKernel = 4;
    constexpr int kCout = 3;
    constexpr int kCin = 2;
    constexpr int kLength = 5;
    constexpr int kStride = 2;
    constexpr int kOutputLen = (kLength - 1) * kStride + kKernel;

    VoxCPMContext graph_ctx(ContextType::Graph, 128, 1024);
    ggml_tensor* weight = graph_ctx.new_tensor_3d(GGML_TYPE_F16, kKernel, kCout, kCin);
    ggml_tensor* input = graph_ctx.new_tensor_3d(GGML_TYPE_F32, kLength, kCin, 1);
    ggml_set_input(input);

    ggml_tensor* output = ggml_conv_transpose_1d(graph_ctx.raw_context(), weight, input, kStride, 0, 1);
    ggml_set_output(output);

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, output);
    backend->reserve_compute_memory(graph, "test.cuda.conv_transpose_1d");
    backend->alloc_graph(graph, "test.cuda.conv_transpose_1d");

    std::vector<float> weight_f32(static_cast<size_t>(kKernel * kCout * kCin), 0.0f);
    for (size_t i = 0; i < weight_f32.size(); ++i) {
        weight_f32[i] = static_cast<float>((static_cast<int>(i) % 7) - 3) * 0.125f;
    }

    std::vector<ggml_fp16_t> weight_f16(weight_f32.size());
    for (size_t i = 0; i < weight_f32.size(); ++i) {
        weight_f16[i] = ggml_fp32_to_fp16(weight_f32[i]);
    }

    std::vector<float> input_data(static_cast<size_t>(kLength * kCin), 0.0f);
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = static_cast<float>((static_cast<int>(i) % 5) - 2) * 0.25f;
    }

    backend->tensor_set(weight, weight_f16.data(), 0, weight_f16.size() * sizeof(ggml_fp16_t));
    backend->tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    REQUIRE(backend->compute(graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(static_cast<size_t>(kOutputLen * kCout), 0.0f);
    backend->tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    REQUIRE(all_finite(actual));

    const std::vector<float> expected =
        conv_transpose_1d_reference(weight_f32, input_data, kKernel, kCout, kCin, kLength, kStride);
    REQUIRE(expected.size() == actual.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        REQUIRE(actual[i] == Approx(expected[i]).margin(1e-4f));
    }
}

}  // namespace test
}  // namespace voxcpm
