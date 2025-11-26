#include <torch/script.h>
#include "jit.h"
#include "exception.hpp"

extern "C"
{

    jit_module jit_load(char **err, const char *path)
    {
        return jit_load_to_device(err, path, 0);
    }

    jit_module jit_load_to_device(char **err, const char *path, int8_t device)
    {
        return auto_catch_jit_module(
            [path, device]()
            {
                torch::Device dev = (device == 0) ? torch::kCPU : torch::kCUDA;
                auto module = new torch::jit::script::Module(torch::jit::load(path, dev));
                module->eval();
                return module;
            },
            err);
    }

    tensor jit_forward(char **err, jit_module m, tensor input)
    {
        return auto_catch_tensor(
            [m, input]()
            {
                if (m == nullptr)
                {
                    throw std::runtime_error("jit_forward: null module");
                }
                if (input == nullptr)
                {
                    throw std::runtime_error("jit_forward: null input tensor");
                }

                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(*input);

                auto output = m->forward(inputs);

                if (output.isTensor())
                {
                    return new torch::Tensor(output.toTensor());
                }
                if (output.isTuple())
                {
                    auto tuple = output.toTuple();
                    return new torch::Tensor(tuple->elements()[0].toTensor());
                }
                throw std::runtime_error("Unexpected output type: expected Tensor or Tuple");
            },
            err);
    }

    size_t jit_forward_multi(char **err, jit_module m, tensor input,
                             tensor *out_tensors, size_t out_count)
    {
        return auto_catch_size_t(
            [m, input, out_tensors, out_count]() -> size_t
            {
                if (m == nullptr)
                {
                    throw std::runtime_error("jit_forward_multi: null module");
                }
                if (input == nullptr)
                {
                    throw std::runtime_error("jit_forward_multi: null input tensor");
                }
                if (out_count > 0 && out_tensors == nullptr)
                {
                    throw std::runtime_error("jit_forward_multi: null out_tensors for nonzero out_count");
                }

                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(*input);

                auto output = m->forward(inputs);

                if (output.isTuple())
                {
                    auto tuple = output.toTuple();
                    size_t actual_count = std::min(tuple->elements().size(), out_count);
                    for (size_t i = 0; i < actual_count; i++)
                    {
                        out_tensors[i] = new torch::Tensor(tuple->elements()[i].toTensor());
                    }
                    return actual_count;
                }

                if (output.isTensor())
                {
                    if (out_count >= 1)
                    {
                        out_tensors[0] = new torch::Tensor(output.toTensor());
                        return 1;
                    }
                    return 0;
                }

                throw std::runtime_error("Unexpected output type: expected Tensor or Tuple");
            },
            err);
    }

    void jit_to_device(char **err, jit_module m, int8_t device)
    {
        auto_catch_void(
            [m, device]()
            {
                if (m == nullptr)
                {
                    throw std::runtime_error("jit_to_device: null module");
                }
                torch::Device dev = (device == 0) ? torch::kCPU : torch::kCUDA;
                m->to(dev);
            },
            err);
    }

    void jit_eval(jit_module m)
    {
        if (m != nullptr)
        {
            m->eval();
        }
    }

    void jit_train(jit_module m)
    {
        if (m != nullptr)
        {
            m->train();
        }
    }

    void jit_free(jit_module m)
    {
        if (m != nullptr)
        {
            delete m;
        }
    }

} // extern "C"
