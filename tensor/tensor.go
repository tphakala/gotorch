package tensor

import (
	"runtime"
	"sync"
	"time"

	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/internal/torch"
	"github.com/lwch/logging"
)

// tensorHandle wraps the tensor with once-only cleanup semantics.
type tensorHandle struct {
	t    torch.Tensor
	idx  uint64
	once sync.Once
}

func (h *tensorHandle) free() {
	h.once.Do(func() {
		if h.t != nil {
			logging.Debug("free tensor: %d", h.idx)
			freeLeakTracking(h.idx)
			torch.FreeTensor(h.t)
			h.t = nil
		}
	})
}

type Tensor struct {
	idx     uint64
	created time.Time
	handle  *tensorHandle
	t       torch.Tensor
}

func New(t torch.Tensor) *Tensor {
	ts := &Tensor{
		created: time.Now(),
		t:       t,
	}
	logBuildInfo(ts)
	logging.Debug("new tensor: %d", ts.idx)

	// Go 1.24+: Use AddCleanup instead of SetFinalizer
	// Benefits: no cycle leaks, multiple cleanups allowed, parallel execution
	handle := &tensorHandle{t: t, idx: ts.idx}
	ts.handle = handle
	runtime.AddCleanup(ts, func(h *tensorHandle) {
		h.free()
	}, handle)

	return ts
}

func (t *Tensor) Created() time.Time {
	return t.created
}

func (t *Tensor) Tensor() torch.Tensor {
	return t.t
}

func (t *Tensor) Reshape(shape ...int64) *Tensor {
	ptr := torch.Reshape(t.t, shape)
	return New(ptr)
}

func (t *Tensor) Transpose(dim1, dim2 int64) *Tensor {
	ptr := torch.Transpose(t.t, dim1, dim2)
	return New(ptr)
}

func (t *Tensor) ElemSize() int64 {
	return torch.ElemSize(t.t)
}

func (t *Tensor) ElemCount() int64 {
	return torch.ElemCount(t.t)
}

func (t *Tensor) Dims() int64 {
	return torch.Dims(t.t)
}

func (t *Tensor) Shapes() []int64 {
	return torch.Shapes(t.t)
}

func (t *Tensor) ScalarType() consts.ScalarType {
	return torch.ScalarType(t.t)
}

func (t *Tensor) DeviceType() consts.DeviceType {
	return torch.DeviceType(t.t)
}

func (t *Tensor) SetRequiresGrad(b bool) {
	torch.SetRequiresGrad(t.t, b)
}

func (t *Tensor) ToDevice(device consts.DeviceType) *Tensor {
	ptr := torch.ToDevice(t.t, device)
	return New(ptr)
}

func (t *Tensor) ToScalarType(scalarType consts.ScalarType) *Tensor {
	ptr := torch.ToScalarType(t.t, scalarType)
	return New(ptr)
}

func (t *Tensor) Detach() *Tensor {
	ptr := torch.Detach(t.t)
	return New(ptr)
}

func (t *Tensor) Clone() *Tensor {
	ptr := torch.Clone(t.t)
	return New(ptr)
}
