package torch

import (
	"fmt"

	"github.com/lwch/gotorch/internal/model/storage"
	"github.com/nlpodyssey/gopickle/types"
)

// https://github.com/pytorch/pytorch/blob/main/torch/_utils.py

type RebuildTensorV2 struct{}

var _ types.Callable = &RebuildTensorV2{}

func (r *RebuildTensorV2) Call(args ...any) (any, error) {
	if len(args) != 6 {
		return nil, fmt.Errorf("RebuildTensorV2 unexpected args: %#v", args)
	}

	stor, storageOk := args[0].(storage.Storage)
	size, sizeOk := args[2].(*types.Tuple)
	requiresGrad, requiresGradOk := args[4].(bool)
	if !storageOk || !sizeOk || !requiresGradOk {
		return nil, fmt.Errorf("RebuildTensorV2 unexpected args: %#v", args)
	}

	shape := tupleToInt64Slice(size)

	stor.SetShape(shape)
	stor.SetRequiresGrad(requiresGrad)

	return stor, nil
}

func tupleToInt64Slice(tuple *types.Tuple) []int64 {
	length := tuple.Len()
	slice := make([]int64, length)
	for i := range length {
		value, ok := tuple.Get(i).(int)
		if !ok {
			fmt.Printf("WARNING: tuple of ints expected. Got %#v\n", tuple)
			continue
		}
		slice[i] = int64(value)
	}
	return slice
}
