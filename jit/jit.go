// Package jit provides TorchScript model loading and inference capabilities.
package jit

import (
	"errors"
	"fmt"
	"iter"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/internal/torch"
	"github.com/lwch/gotorch/tensor"
)

// ErrModuleClosed is returned when operations are attempted on a closed module.
var ErrModuleClosed = errors.New("module has been closed")

// moduleHandle wraps the JIT module with once-only cleanup semantics.
type moduleHandle struct {
	m      torch.JitModule
	once   sync.Once
	closed atomic.Bool
}

func (h *moduleHandle) free() {
	h.once.Do(func() {
		h.closed.Store(true)
		torch.JitFree(h.m)
	})
}

func (h *moduleHandle) isClosed() bool {
	return h.closed.Load()
}

// Module represents a loaded TorchScript model.
type Module struct {
	handle *moduleHandle
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

	jm := torch.JitLoadToDevice(path, device)
	handle := &moduleHandle{m: jm}
	module := &Module{handle: handle}

	// Go 1.24+: Use AddCleanup instead of SetFinalizer
	// Benefits: no cycle leaks, multiple cleanups allowed, works with interior pointers
	// The sync.Once in moduleHandle ensures free is called exactly once
	runtime.AddCleanup(module, func(h *moduleHandle) {
		h.free()
	}, handle)

	return module, nil
}

// Close explicitly releases module resources.
// The module should not be used after calling Close.
// Safe to call multiple times.
func (m *Module) Close() {
	if m != nil && m.handle != nil {
		m.handle.free()
	}
}

// Forward runs inference and returns a single output tensor.
// For models returning multiple outputs, this returns only the first.
func (m *Module) Forward(input *tensor.Tensor) (out *tensor.Tensor, err error) {
	if m == nil || m.handle == nil {
		return nil, ErrModuleClosed
	}
	if m.handle.isClosed() {
		return nil, ErrModuleClosed
	}

	defer func() {
		if r := recover(); r != nil {
			out = nil
			err = fmt.Errorf("forward pass failed: %v", r)
		}
	}()

	result := torch.JitForward(m.handle.m, input.Tensor())
	return tensor.New(result), nil
}

// ForwardMulti runs inference and returns multiple output tensors.
// This is useful for models like BirdNET v3.0 that return (embeddings, predictions).
func (m *Module) ForwardMulti(input *tensor.Tensor, numOutputs int) (out []*tensor.Tensor, err error) {
	if m == nil || m.handle == nil {
		return nil, ErrModuleClosed
	}
	if m.handle.isClosed() {
		return nil, ErrModuleClosed
	}

	defer func() {
		if r := recover(); r != nil {
			out = nil
			err = fmt.Errorf("forward pass failed: %v", r)
		}
	}()

	outputs := torch.JitForwardMulti(m.handle.m, input.Tensor(), numOutputs)
	result := make([]*tensor.Tensor, len(outputs))
	for i, t := range outputs {
		result[i] = tensor.New(t)
	}
	return result, nil
}

// ToDevice moves the module to the specified device.
func (m *Module) ToDevice(device consts.DeviceType) (err error) {
	if m == nil || m.handle == nil {
		return ErrModuleClosed
	}
	if m.handle.isClosed() {
		return ErrModuleClosed
	}

	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("failed to move module to device: %v", r)
		}
	}()

	torch.JitToDevice(m.handle.m, device)
	return nil
}

// Eval sets the module to evaluation mode.
// This disables dropout and batch normalization updates.
func (m *Module) Eval() {
	if m == nil || m.handle == nil || m.handle.isClosed() {
		return
	}
	torch.JitEval(m.handle.m)
}

// Train sets the module to training mode.
func (m *Module) Train() {
	if m == nil || m.handle == nil || m.handle.isClosed() {
		return
	}
	torch.JitTrain(m.handle.m)
}

// Outputs returns an iterator over the output tensors from ForwardMulti.
// This leverages Go 1.23+ iterator support for cleaner code.
func Outputs(tensors []*tensor.Tensor) iter.Seq2[int, *tensor.Tensor] {
	return func(yield func(int, *tensor.Tensor) bool) {
		for i, t := range tensors {
			if !yield(i, t) {
				return
			}
		}
	}
}
