package torch

import (
	"runtime"
	"sync"
	"time"
	"unsafe"
)

// #include "optimizer.h"
import "C"

// optimizerHandle wraps the optimizer with once-only cleanup semantics.
type optimizerHandle struct {
	data C.optimizer
	once sync.Once
}

func (h *optimizerHandle) free() {
	h.once.Do(func() {
		if h.data != nil {
			var err *C.char
			C.free_optimizer(&err, h.data)
			h.data = nil
		}
	})
}

type Optimizer struct {
	m      sync.Mutex
	handle *optimizerHandle
	data   C.optimizer
}

func NewAdamOptimizer(params []Tensor, lr, beta1, beta2, eps, weightDecay float64) *Optimizer {
	list := make([]C.tensor, len(params))
	for i, p := range params {
		list[i] = C.tensor(p)
	}
	var err *C.char
	ptr := C.new_adam_optimizer(&err, (*C.tensor)(unsafe.Pointer(&list[0])), C.size_t(len(params)), C.double(lr), C.double(beta1), C.double(beta2), C.double(eps), C.double(weightDecay))
	if err != nil {
		panic(C.GoString(err))
	}
	handle := &optimizerHandle{data: ptr}
	optm := &Optimizer{handle: handle, data: ptr}
	runtime.AddCleanup(optm, func(h *optimizerHandle) {
		h.free()
	}, handle)
	return optm
}

func NewAdamWOptimizer(params []Tensor, lr, beta1, beta2, eps, weightDecay float64, amsgrad bool) *Optimizer {
	list := make([]C.tensor, len(params))
	for i, p := range params {
		list[i] = C.tensor(p)
	}
	var err *C.char
	ptr := C.new_adamw_optimizer(&err, (*C.tensor)(unsafe.Pointer(&list[0])), C.size_t(len(params)), C.double(lr), C.double(beta1), C.double(beta2), C.double(eps), C.bool(amsgrad), C.double(weightDecay))
	if err != nil {
		panic(C.GoString(err))
	}
	handle := &optimizerHandle{data: ptr}
	optm := &Optimizer{handle: handle, data: ptr}
	runtime.AddCleanup(optm, func(h *optimizerHandle) {
		h.free()
	}, handle)
	return optm
}

func (optm *Optimizer) Step() {
	optm.m.Lock()
	defer optm.m.Unlock()
	var err *C.char
	C.optimizer_step(&err, optm.data)
	if err != nil {
		panic(C.GoString(err))
	}
}

func (optm *Optimizer) ZeroGrad() {
	optm.m.Lock()
	defer optm.m.Unlock()
	var err *C.char
	C.optimizer_zero_grad(&err, optm.data)
	if err != nil {
		panic(C.GoString(err))
	}
}

func (optm *Optimizer) GetLr() float64 {
	var err *C.char
	lr := C.optimizer_get_lr(&err, optm.data)
	if err != nil {
		panic(C.GoString(err))
	}
	return float64(lr)
}

func (optm *Optimizer) SetLr(lr float64) {
	var err *C.char
	C.optimizer_set_lr(&err, optm.data, C.double(lr))
	if err != nil {
		panic(C.GoString(err))
	}
}

// optimizerStateHandle wraps the optimizer state with once-only cleanup semantics.
type optimizerStateHandle struct {
	data C.optimizer_state
	once sync.Once
}

func (h *optimizerStateHandle) free() {
	h.once.Do(func() {
		if h.data != nil {
			C.optimizer_state_free(h.data)
			h.data = nil
		}
	})
}

type OptimizerState struct {
	created time.Time
	handle  *optimizerStateHandle
	data    C.optimizer_state
}

func (optm *Optimizer) GetState() *OptimizerState {
	var err *C.char
	data := C.optimizer_get_state(&err, optm.data)
	if err != nil {
		panic(C.GoString(err))
	}
	handle := &optimizerStateHandle{data: data}
	os := &OptimizerState{
		created: time.Now(),
		handle:  handle,
		data:    data,
	}
	runtime.AddCleanup(os, func(h *optimizerStateHandle) {
		h.free()
	}, handle)
	return os
}

func (os *OptimizerState) Size() int {
	var err *C.char
	size := C.optimizer_state_count(&err, os.data)
	if err != nil {
		panic(C.GoString(err))
	}
	return int(size)
}

func (os *OptimizerState) Get(index int) []Tensor {
	var err *C.char
	size := C.optimizer_state_size(&err, os.data, C.size_t(index))
	if err != nil {
		panic(C.GoString(err))
	}
	tensors := make([]Tensor, 0, int(size))
	for i := range int(size) {
		var err *C.char
		tensor := C.optimizer_state_get(&err, os.data, C.size_t(index), C.size_t(i))
		if err != nil {
			panic(C.GoString(err))
		}
		tensors = append(tensors, Tensor(tensor))
	}
	return tensors
}

func (os *OptimizerState) Set(index int, values []Tensor) {
	var err *C.char
	size := C.optimizer_state_size(&err, os.data, C.size_t(index))
	if err != nil {
		panic(C.GoString(err))
	}
	if size != 0 && len(values) != int(size) {
		panic("invalid size")
	}
	for i := range int(size) {
		var err *C.char
		C.optimizer_state_set(&err, os.data, C.size_t(index), C.size_t(i), C.tensor(values[i]))
		if err != nil {
			panic(C.GoString(err))
		}
	}
}
