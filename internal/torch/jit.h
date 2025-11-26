#ifndef __GOTORCH_JIT_H__
#define __GOTORCH_JIT_H__

#include <stddef.h>
#include <stdint.h>
#include "api.h"

#ifdef __cplusplus
extern "C"
{
#endif

    // Load TorchScript model from file (defaults to CPU)
    GOTORCH_API jit_module jit_load(char **err, const char *path);

    // Load TorchScript model to specified device (0=CPU, 1=CUDA)
    GOTORCH_API jit_module jit_load_to_device(char **err, const char *path, int8_t device);

    // Forward pass with single output (returns first tensor if tuple)
    GOTORCH_API tensor jit_forward(char **err, jit_module m, tensor input);

    // Forward pass with multiple outputs (for models returning tuples)
    // out_tensors: pre-allocated array, out_count: array size
    // Returns: actual number of outputs written
    GOTORCH_API size_t jit_forward_multi(char **err, jit_module m, tensor input,
                                          tensor *out_tensors, size_t out_count);

    // Move module to device
    GOTORCH_API void jit_to_device(char **err, jit_module m, int8_t device);

    // Set evaluation mode (disables dropout, batch norm updates)
    GOTORCH_API void jit_eval(jit_module m);

    // Set training mode
    GOTORCH_API void jit_train(jit_module m);

    // Free module resources
    GOTORCH_API void jit_free(jit_module m);

#ifdef __cplusplus
}
#endif

#endif // __GOTORCH_JIT_H__
