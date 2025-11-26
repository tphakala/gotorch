package torch

// #cgo noescape jit_forward
// #cgo noescape jit_forward_multi
// #cgo nocallback jit_forward
// #cgo nocallback jit_forward_multi
// #cgo nocallback jit_eval
// #cgo nocallback jit_train
// #cgo nocallback jit_free
// #include <stdlib.h>
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
