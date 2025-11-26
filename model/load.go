package model

import (
	"sync"

	"github.com/lwch/gotorch/internal/model"
	"github.com/lwch/gotorch/internal/model/storage"
	"github.com/lwch/gotorch/tensor"
)

type Model struct {
	params map[string]*tensor.Tensor
}

func Load(dir string) (*Model, error) {
	m, err := model.Load(dir)
	if err != nil {
		return nil, err
	}
	var mu sync.Mutex
	params := make(map[string]*tensor.Tensor)
	var wg sync.WaitGroup
	wg.Add(len(m.Params()))
	for k, v := range m.Params() {
		go func(k string, v storage.Storage) {
			defer wg.Done()
			t := buildTensor(v)
			mu.Lock()
			params[k] = t
			mu.Unlock()
		}(k, v)
	}
	wg.Wait()
	return &Model{params: params}, nil
}

func buildTensor(t storage.Storage) *tensor.Tensor {
	shape := t.GetShape()
	switch t.Type() {
	// float to bfloat16 tensor
	case storage.TypeBFloat16:
		data, ok := t.Get().([]uint16)
		if !ok {
			panic("buildTensor: TypeBFloat16 storage does not contain []uint16")
		}
		return tensor.FromBFloat16Raw(data, tensor.WithShapes(shape...))
	// float to half tensor
	case storage.TypeHalf:
		data, ok := t.Get().([]uint16)
		if !ok {
			panic("buildTensor: TypeHalf storage does not contain []uint16")
		}
		return tensor.FromHalfRaw(data, tensor.WithShapes(shape...))
	// float to float32 tensor
	case storage.TypeFloat:
		data, ok := t.Get().([]float32)
		if !ok {
			panic("buildTensor: TypeFloat storage does not contain []float32")
		}
		return tensor.FromFloat32(data, tensor.WithShapes(shape...))
	// double to float64 tensor
	case storage.TypeDouble:
		data, ok := t.Get().([]float64)
		if !ok {
			panic("buildTensor: TypeDouble storage does not contain []float64")
		}
		return tensor.FromFloat64(data, tensor.WithShapes(shape...))
	// byte to uint8 tensor
	case storage.TypeByte:
		data, ok := t.Get().([]byte)
		if !ok {
			panic("buildTensor: TypeByte storage does not contain []byte")
		}
		return tensor.FromUint8(data, tensor.WithShapes(shape...))
	// char to int8 tensor
	case storage.TypeChar:
		data, ok := t.Get().([]int8)
		if !ok {
			panic("buildTensor: TypeChar storage does not contain []int8")
		}
		return tensor.FromInt8(data, tensor.WithShapes(shape...))
	// short to int16 tensor
	case storage.TypeShort:
		data, ok := t.Get().([]int16)
		if !ok {
			panic("buildTensor: TypeShort storage does not contain []int16")
		}
		return tensor.FromInt16(data, tensor.WithShapes(shape...))
	// int to int32 tensor
	case storage.TypeInt:
		data, ok := t.Get().([]int32)
		if !ok {
			panic("buildTensor: TypeInt storage does not contain []int32")
		}
		return tensor.FromInt32(data, tensor.WithShapes(shape...))
	// long to int64 tensor
	case storage.TypeLong:
		data, ok := t.Get().([]int64)
		if !ok {
			panic("buildTensor: TypeLong storage does not contain []int64")
		}
		return tensor.FromInt64(data, tensor.WithShapes(shape...))
	default:
		panic("not supported")
	}
}

func (m *Model) Get(name string) *tensor.Tensor {
	return m.params[name]
}

func (m *Model) Params() map[string]*tensor.Tensor {
	return m.params
}
