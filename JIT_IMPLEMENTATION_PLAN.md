# TorchScript/JIT Implementation Plan for gotorch

> **Status:** Implementation Complete
> **Go Version:** 1.25 (minimum)
> **Last Updated:** November 2025

## Overview

This document outlines the implementation plan for adding TorchScript (JIT) support to gotorch, enabling loading and inference of pre-compiled PyTorch models like BirdNET v3.0.

### Target Model Specification (BirdNET v3.0)

| Property | Value |
|----------|-------|
| Model Type | RecursiveScriptModule (TorchScript) |
| Parameters | 20,868,750 (~80MB FP32) |
| Input | `[batch, 160000]` float32 (5s @ 32kHz PCM) |
| Output[0] | `[batch, 1280]` float32 - Embeddings |
| Output[1] | `[batch, 1225]` float32 - Predictions (species logits) |

---

## Part 1: Implementation Architecture

### File Structure

```
gotorch/
├── internal/torch/
│   ├── api.h          # MODIFY: Add jit_module typedef
│   ├── exception.hpp  # MODIFY: Add auto_catch_jit_module template
│   ├── jit.h          # NEW: C function declarations
│   └── jit.go         # NEW: Low-level CGO bindings
├── lib/
│   ├── CMakeLists.txt # MODIFY: Add jit.cpp to build
│   └── jit.cpp        # NEW: C++ implementation
└── jit/
    └── jit.go         # NEW: High-level public API
```

### Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│  User Code                                              │
│  model, _ := jit.Load("model.pt")                       │
│  output, _ := model.Forward(input)                      │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  jit/jit.go (High-level API)                            │
│  - Error handling via panic recovery                    │
│  - Returns *tensor.Tensor                               │
│  - Manages finalizers                                   │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  internal/torch/jit.go (Low-level CGO)                  │
│  - Raw CGO bindings                                     │
│  - Panic on error (matches existing pattern)            │
│  - Works with torch.Tensor (C.tensor)                   │
└────────────────────────┬────────────────────────────────┘
                         │ CGO
┌────────────────────────▼────────────────────────────────┐
│  lib/jit.cpp (C++ Implementation)                       │
│  - Uses exception.hpp templates                         │
│  - Interfaces with torch::jit::script::Module           │
└─────────────────────────────────────────────────────────┘
```

---

## Part 2: Detailed Implementation

### 2.1 Modify `internal/torch/api.h`

```c
#ifndef __GOTORCH_API_H__
#define __GOTORCH_API_H__

#ifdef __cplusplus
#include <torch/torch.h>
#include <torch/script.h>  // ADD: Required for jit::script::Module
extern "C"
{
    typedef torch::Tensor *tensor;
    typedef torch::optim::Optimizer *optimizer;
    typedef torch::nn::Module *module;
    typedef torch::jit::script::Module *jit_module;  // ADD: JIT module type

    struct _optimizer_state
    {
        std::vector<torch::optim::OptimizerParamState *> data;
    };
    typedef _optimizer_state *optimizer_state;
#else
typedef void *tensor;
typedef void *optimizer;
typedef void *module;
typedef void *jit_module;  // ADD: Opaque pointer for C
typedef void *optimizer_state;
#endif

// ... rest unchanged
```

### 2.2 Modify `internal/torch/exception.hpp`

Add new template after existing ones:

```cpp
template <typename Function>
jit_module auto_catch_jit_module(Function f, char **err)
{
    try
    {
        return f();
    }
    catch (const torch::Error &e)
    {
        *err = strdup(e.msg().c_str());
    }
    catch (const std::exception &e)
    {
        *err = strdup(e.what());
    }
    return nullptr;
}
```

### 2.3 Create `internal/torch/jit.h`

```c
#ifndef __GOTORCH_JIT_H__
#define __GOTORCH_JIT_H__

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
```

### 2.4 Create `lib/jit.cpp`

```cpp
#include <torch/script.h>
#include "jit.h"
#include "exception.hpp"

extern "C" {

jit_module jit_load(char **err, const char *path) {
    return jit_load_to_device(err, path, 0);
}

jit_module jit_load_to_device(char **err, const char *path, int8_t device) {
    return auto_catch_jit_module(
        [path, device]() {
            torch::Device dev = (device == 0) ? torch::kCPU : torch::kCUDA;
            auto module = new torch::jit::script::Module(torch::jit::load(path, dev));
            module->eval();
            return module;
        },
        err);
}

tensor jit_forward(char **err, jit_module m, tensor input) {
    return auto_catch_tensor(
        [m, input]() {
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(*input);

            auto output = m->forward(inputs);

            if (output.isTensor()) {
                return new torch::Tensor(output.toTensor());
            }
            if (output.isTuple()) {
                auto tuple = output.toTuple();
                return new torch::Tensor(tuple->elements()[0].toTensor());
            }
            throw std::runtime_error("Unexpected output type: expected Tensor or Tuple");
        },
        err);
}

size_t jit_forward_multi(char **err, jit_module m, tensor input,
                         tensor *out_tensors, size_t out_count) {
    return auto_catch_size_t(
        [m, input, out_tensors, out_count]() -> size_t {
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(*input);

            auto output = m->forward(inputs);

            if (output.isTuple()) {
                auto tuple = output.toTuple();
                size_t actual_count = std::min(tuple->elements().size(), out_count);
                for (size_t i = 0; i < actual_count; i++) {
                    out_tensors[i] = new torch::Tensor(tuple->elements()[i].toTensor());
                }
                return actual_count;
            }

            if (output.isTensor()) {
                if (out_count >= 1) {
                    out_tensors[0] = new torch::Tensor(output.toTensor());
                    return 1;
                }
                return 0;
            }

            throw std::runtime_error("Unexpected output type: expected Tensor or Tuple");
        },
        err);
}

void jit_to_device(char **err, jit_module m, int8_t device) {
    auto_catch_void(
        [m, device]() {
            torch::Device dev = (device == 0) ? torch::kCPU : torch::kCUDA;
            m->to(dev);
        },
        err);
}

void jit_eval(jit_module m) {
    if (m != nullptr) {
        m->eval();
    }
}

void jit_train(jit_module m) {
    if (m != nullptr) {
        m->train();
    }
}

void jit_free(jit_module m) {
    if (m != nullptr) {
        delete m;
    }
}

} // extern "C"
```

### 2.5 Create `internal/torch/jit.go`

```go
package torch

// #include "jit.h"
import "C"
import (
	"unsafe"

	"github.com/lwch/gotorch/consts"
)

// JitModule is the low-level handle to a TorchScript module
type JitModule C.jit_module

// JitLoad loads a TorchScript model from file (CPU)
func JitLoad(path string) JitModule {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	var err *C.char
	m := C.jit_load(&err, cpath)
	if err != nil {
		defer C.free(unsafe.Pointer(err))
		panic(C.GoString(err))
	}
	return JitModule(m)
}

// JitLoadToDevice loads a TorchScript model to specified device
func JitLoadToDevice(path string, device consts.DeviceType) JitModule {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	var err *C.char
	m := C.jit_load_to_device(&err, cpath, C.int8_t(device))
	if err != nil {
		defer C.free(unsafe.Pointer(err))
		panic(C.GoString(err))
	}
	return JitModule(m)
}

// JitForward runs forward pass with single output
func JitForward(m JitModule, input Tensor) Tensor {
	var err *C.char
	out := C.jit_forward(&err, C.jit_module(m), C.tensor(input))
	if err != nil {
		defer C.free(unsafe.Pointer(err))
		panic(C.GoString(err))
	}
	return Tensor(out)
}

// JitForwardMulti runs forward pass returning multiple outputs
func JitForwardMulti(m JitModule, input Tensor, numOutputs int) []Tensor {
	if numOutputs <= 0 {
		return nil
	}

	outputs := make([]C.tensor, numOutputs)

	var err *C.char
	actual := C.jit_forward_multi(&err, C.jit_module(m), C.tensor(input),
		&outputs[0], C.size_t(numOutputs))
	if err != nil {
		defer C.free(unsafe.Pointer(err))
		panic(C.GoString(err))
	}

	result := make([]Tensor, actual)
	for i := range int(actual) {
		result[i] = Tensor(outputs[i])
	}
	return result
}

// JitToDevice moves module to specified device
func JitToDevice(m JitModule, device consts.DeviceType) {
	var err *C.char
	C.jit_to_device(&err, C.jit_module(m), C.int8_t(device))
	if err != nil {
		defer C.free(unsafe.Pointer(err))
		panic(C.GoString(err))
	}
}

// JitEval sets module to evaluation mode
func JitEval(m JitModule) {
	C.jit_eval(C.jit_module(m))
}

// JitTrain sets module to training mode
func JitTrain(m JitModule) {
	C.jit_train(C.jit_module(m))
}

// JitFree releases module resources
func JitFree(m JitModule) {
	C.jit_free(C.jit_module(m))
}
```

### 2.6 Create `jit/jit.go`

```go
// Package jit provides TorchScript model loading and inference capabilities.
package jit

import (
	"fmt"
	"runtime"

	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/internal/torch"
	"github.com/lwch/gotorch/tensor"
)

// Module represents a loaded TorchScript model.
type Module struct {
	m torch.JitModule
}

// Load loads a TorchScript model from file to CPU.
func Load(path string) (m *Module, err error) {
	return LoadToDevice(path, consts.KCPU)
}

// LoadToDevice loads a TorchScript model to the specified device.
func LoadToDevice(path string, device consts.DeviceType) (m *Module, err error) {
	defer func() {
		if r := recover(); r != nil {
			m = nil
			err = fmt.Errorf("failed to load model %q: %v", path, r)
		}
	}()

	module := &Module{m: torch.JitLoadToDevice(path, device)}
	runtime.SetFinalizer(module, freeModule)
	return module, nil
}

func freeModule(m *Module) {
	if m != nil && m.m != nil {
		torch.JitFree(m.m)
		m.m = nil
	}
}

// Close explicitly releases module resources.
// The module should not be used after calling Close.
func (m *Module) Close() {
	freeModule(m)
	runtime.SetFinalizer(m, nil)
}

// Forward runs inference and returns a single output tensor.
// For models returning multiple outputs, this returns only the first.
func (m *Module) Forward(input *tensor.Tensor) (out *tensor.Tensor, err error) {
	defer func() {
		if r := recover(); r != nil {
			out = nil
			err = fmt.Errorf("forward pass failed: %v", r)
		}
	}()

	result := torch.JitForward(m.m, input.Tensor())
	return tensor.New(result), nil
}

// ForwardMulti runs inference and returns multiple output tensors.
// This is useful for models like BirdNET v3.0 that return (embeddings, predictions).
func (m *Module) ForwardMulti(input *tensor.Tensor, numOutputs int) (out []*tensor.Tensor, err error) {
	defer func() {
		if r := recover(); r != nil {
			out = nil
			err = fmt.Errorf("forward pass failed: %v", r)
		}
	}()

	outputs := torch.JitForwardMulti(m.m, input.Tensor(), numOutputs)
	result := make([]*tensor.Tensor, len(outputs))
	for i, t := range outputs {
		result[i] = tensor.New(t)
	}
	return result, nil
}

// ToDevice moves the module to the specified device.
func (m *Module) ToDevice(device consts.DeviceType) error {
	defer func() {
		if r := recover(); r != nil {
			// Error is set via named return in actual implementation
		}
	}()

	var err error
	func() {
		defer func() {
			if r := recover(); r != nil {
				err = fmt.Errorf("failed to move module to device: %v", r)
			}
		}()
		torch.JitToDevice(m.m, device)
	}()
	return err
}

// Eval sets the module to evaluation mode.
// This disables dropout and batch normalization updates.
func (m *Module) Eval() {
	torch.JitEval(m.m)
}

// Train sets the module to training mode.
func (m *Module) Train() {
	torch.JitTrain(m.m)
}
```

### 2.7 Modify `lib/CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(gotorch)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_definitions(-DGOTORCH_EXPORT)

include_directories(../internal/torch)

add_library(gotorch SHARED
	../internal/torch/api.h ../internal/torch/exception.hpp
	../internal/torch/loss.h loss.cpp
	../internal/torch/operator.h operator.cpp
	../internal/torch/optimizer.h optimizer.cpp
	../internal/torch/module.h module.cpp
	../internal/torch/tensor.h tensor.cpp
	../internal/torch/jit.h jit.cpp
	conv.cpp
	utils.cpp)
target_link_libraries(gotorch "${TORCH_LIBRARIES}")
set_property(TARGET gotorch PROPERTY CXX_STANDARD 17)
```

---

## Part 3: Go 1.25 Features Utilized

### Minimum Version: Go 1.25

The gotorch codebase now requires **Go 1.25** as minimum. All features from Go 1.23, 1.24, and 1.25 are used directly without build tags.

### Go 1.23 Features Used

| Feature | Current Pattern | New Pattern | Benefit |
|---------|-----------------|-------------|---------|
| **Range over int** | `for i := 0; i < n; i++` | `for i := range n` | Cleaner iteration |
| **Iterator functions** | Manual slice iteration | `iter.Seq[T]` patterns | Composable iteration |
| **`unique.Handle[T]`** | N/A | Canonicalize tensor metadata | Memory optimization |

#### 3.1 Range Over Integer (Go 1.22+, stable in 1.23)

**Before:**
```go
for i := 0; i < int(actual); i++ {
    result[i] = Tensor(outputs[i])
}
```

**After:**
```go
for i := range int(actual) {
    result[i] = Tensor(outputs[i])
}
```

#### 3.2 Iterator Support for Tensor Collections

Add iterator methods to tensor slices:

```go
// In tensor package
import "iter"

// Values returns an iterator over tensor values
func Values(tensors []*Tensor) iter.Seq[*Tensor] {
    return func(yield func(*Tensor) bool) {
        for _, t := range tensors {
            if !yield(t) {
                return
            }
        }
    }
}

// Usage with ForwardMulti
outputs, _ := model.ForwardMulti(input, 2)
for t := range tensor.Values(outputs) {
    fmt.Println(t.Shapes())
}
```

### Go 1.24 Features Used

| Feature | Application | Benefit |
|---------|-------------|---------|
| **`runtime.AddCleanup`** | Replace `SetFinalizer` for tensors/modules | No cycle leaks, multiple cleanups |
| **`weak.Pointer[T]`** | Tensor caching, model registry | Memory-efficient caches |
| **CGO `#cgo noescape`** | JIT forward calls | Reduced GC pressure |
| **CGO `#cgo nocallback`** | Most C functions | Better optimization |

#### 3.3 Replace SetFinalizer with AddCleanup (Go 1.24+)

**Before:**
```go
func New(t torch.Tensor) *Tensor {
    ts := &Tensor{t: t}
    runtime.SetFinalizer(ts, freeTensor)
    return ts
}
```

**After:**
```go
func New(t torch.Tensor) *Tensor {
    ts := &Tensor{t: t}
    runtime.AddCleanup(ts, func(t torch.Tensor) {
        torch.FreeTensor(t)
    }, t)
    return ts
}
```

**Benefits:**
- Multiple cleanups per object (can track both tensor and metadata)
- Works with interior pointers
- No cycle-induced memory leaks
- Cleaner semantics

#### 3.4 CGO Escape Analysis Hints

Add to `internal/torch/jit.go`:

```go
// #cgo noescape jit_forward
// #cgo noescape jit_forward_multi
// #cgo nocallback jit_forward
// #cgo nocallback jit_forward_multi
// #cgo nocallback jit_eval
// #cgo nocallback jit_train
// #include "jit.h"
import "C"
```

**Benefits:**
- Compiler knows memory doesn't escape to C
- Reduces heap allocations
- Better performance for hot paths

#### 3.5 Weak Pointers for Model Caching

```go
import "weak"

// ModelCache provides weak-reference caching for loaded models
type ModelCache struct {
    mu    sync.RWMutex
    cache map[string]weak.Pointer[Module]
}

func (c *ModelCache) Get(path string) *Module {
    c.mu.RLock()
    if wp, ok := c.cache[path]; ok {
        if m := wp.Value(); m != nil {
            c.mu.RUnlock()
            return m
        }
    }
    c.mu.RUnlock()
    return nil
}

func (c *ModelCache) Set(path string, m *Module) {
    c.mu.Lock()
    c.cache[path] = weak.Make(m)
    c.mu.Unlock()
}
```

### Go 1.25 Features Available

| Feature | Application | Benefit |
|---------|-------------|---------|
| **`testing/synctest`** | Test concurrent inference | Deterministic timing tests |
| **`runtime/trace.FlightRecorder`** | Debug inference issues | Low-overhead tracing |
| **Container-aware GOMAXPROCS** | Kubernetes deployments | Automatic CPU limit respect |
| **Parallel cleanups** | High tensor throughput | Faster cleanup execution |

#### 3.6 Flight Recorder for Debugging

```go
import "runtime/trace"

var recorder *trace.FlightRecorder

func init() {
    recorder = trace.NewFlightRecorder()
    recorder.Start()
}

// Call when inference fails unexpectedly
func DumpTrace(w io.Writer) error {
    _, err := recorder.WriteTo(w)
    return err
}
```

#### 3.7 Synctest for Concurrent Inference Tests

```go
import "testing/synctest"

func TestConcurrentInference(t *testing.T) {
    synctest.Test(t, func(t *testing.T) {
        model, _ := jit.Load("model.pt")
        defer model.Close()

        var wg sync.WaitGroup
        for i := range 10 {
            wg.Add(1)
            go func(id int) {
                defer wg.Done()
                input := tensor.Zeros([]int64{1, 160000}, consts.KFloat, consts.KCPU)
                _, err := model.Forward(input)
                if err != nil {
                    t.Errorf("inference %d failed: %v", id, err)
                }
            }(i)
        }
        synctest.Wait() // Wait for all goroutines to block or complete
        wg.Wait()
    })
}
```

---

## Part 4: Implementation Checklist

### Phase 1: Core Implementation (COMPLETE)

- [x] Modify `internal/torch/api.h` - add jit_module typedef
- [x] Modify `internal/torch/exception.hpp` - add auto_catch_jit_module
- [x] Create `internal/torch/jit.h` - C function declarations
- [x] Create `lib/jit.cpp` - C++ implementation
- [x] Modify `lib/CMakeLists.txt` - add jit.cpp
- [ ] Rebuild libgotorch.so

### Phase 2: Go Bindings (COMPLETE)

- [x] Create `internal/torch/jit.go` - low-level CGO bindings
- [x] Create `jit/jit.go` - high-level public API
- [x] Add CGO escape hints (`#cgo noescape`, `#cgo nocallback`)

### Phase 3: Testing

- [ ] Create `jit/jit_test.go` - unit tests
- [ ] Test with simple TorchScript model
- [ ] Test with BirdNET v3.0 model
- [ ] Test CPU and CUDA paths
- [ ] Test multi-output inference

### Phase 4: Go 1.25 Modernization (COMPLETE)

- [x] Update go.mod to require Go 1.25
- [x] Use `runtime.AddCleanup` instead of `SetFinalizer` (jit/jit.go)
- [x] Add iterator method `Outputs()` (jit/jit.go)
- [x] Add CGO annotations (`#cgo noescape`, `#cgo nocallback`)
- [x] Use `range int(n)` syntax
- [ ] Add weak pointer caching (optional, for future)

### Phase 5: Documentation

- [ ] Update README.md with JIT usage examples
- [x] Add godoc comments to all exported functions
- [ ] Create example/jit/ directory with BirdNET example

---

## Part 5: Usage Example

```go
package main

import (
    "fmt"
    "log"

    "github.com/lwch/gotorch/consts"
    "github.com/lwch/gotorch/jit"
    "github.com/lwch/gotorch/tensor"
)

func main() {
    // Load BirdNET v3.0 model to GPU
    model, err := jit.LoadToDevice("BirdNET_V3.0.pt", consts.KCUDA)
    if err != nil {
        log.Fatal(err)
    }
    defer model.Close()

    // Create input: 5 seconds of audio at 32kHz
    audioData := make([]float32, 160000)
    // ... fill with actual audio samples ...

    input := tensor.FromFloat32(audioData,
        tensor.WithShapes(1, 160000),
        tensor.WithDevice(consts.KCUDA))

    // Run inference - returns embeddings and predictions
    outputs, err := model.ForwardMulti(input, 2)
    if err != nil {
        log.Fatal(err)
    }

    // outputs[0]: embeddings [1, 1280]
    // outputs[1]: predictions [1, 1225]
    embeddings := outputs[0].ToDevice(consts.KCPU).Float32Value()
    predictions := outputs[1].ToDevice(consts.KCPU).Float32Value()

    fmt.Printf("Embeddings: %d dimensions\n", len(embeddings))
    fmt.Printf("Predictions: %d species\n", len(predictions))

    // Find top prediction
    maxIdx, maxVal := 0, float32(-1e9)
    for i, v := range predictions {
        if v > maxVal {
            maxIdx, maxVal = i, v
        }
    }
    fmt.Printf("Top species: index %d (score %.4f)\n", maxIdx, maxVal)
}
```

---

## Part 6: Future Enhancements

### Potential Additional APIs

```go
// Get model metadata
func (m *Module) GetMethod(name string) (*Method, error)
func (m *Module) Methods() []string
func (m *Module) Attributes() map[string]interface{}

// Batch inference
func (m *Module) ForwardBatch(inputs []*tensor.Tensor) ([]*tensor.Tensor, error)

// Tracing support
func Trace(module interface{}, exampleInputs ...*tensor.Tensor) (*Module, error)

// Model optimization
func (m *Module) Optimize() error  // torch::jit::optimize_for_inference
func (m *Module) Freeze() error    // torch::jit::freeze
```

### Performance Optimizations

1. **Input tensor pooling** - Reuse input tensors for batch processing
2. **Output tensor pre-allocation** - Avoid repeated allocations in hot loops
3. **CUDA stream support** - Enable async inference on multiple streams
4. **TensorRT integration** - For maximum inference performance on NVIDIA GPUs

---

## Appendix: Go 1.25 Features Summary

All features below are available with Go 1.25 (the minimum required version):

| Feature | Used In | Status |
|---------|---------|--------|
| Range over int | `internal/torch/jit.go` | ✓ Used |
| Iterator functions (`iter.Seq2`) | `jit/jit.go` | ✓ Used |
| `runtime.AddCleanup` | `jit/jit.go` | ✓ Used |
| CGO `#cgo noescape` | `internal/torch/jit.go` | ✓ Used |
| CGO `#cgo nocallback` | `internal/torch/jit.go` | ✓ Used |
| `weak.Pointer[T]` | Model caching | Available |
| `testing/synctest` | Unit tests | Available |
| `runtime/trace.FlightRecorder` | Debugging | Available |
| Parallel cleanups | Runtime | Automatic |
| Container-aware GOMAXPROCS | Kubernetes | Automatic |

**Note:** Go 1.25 is required. No build tags needed - all features used directly.
