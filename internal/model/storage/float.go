package storage

import (
	"archive/zip"
	"encoding/binary"
	"fmt"
	"sync"
)

type Float struct {
	base
	data []float32
}

var _ Storage = &Float{}

func (*Float) New(wg *sync.WaitGroup, size int, file *zip.File) (Storage, error) {
	fs, err := file.Open()
	if err != nil {
		return nil, fmt.Errorf("Float.New: can not open file %s: %w", file.Name, err)
	}
	defer func() { _ = fs.Close() }()
	var ret Float
	ret.data = make([]float32, size)
	wg.Go(func() {
		err = binary.Read(fs, binary.LittleEndian, ret.data)
		if err != nil {
			panic(fmt.Errorf("Float.New: can not read file %s: %w", file.Name, err))
		}
	})
	return &ret, nil
}

func (f *Float) Get() any {
	return f.data
}

func (*Float) Type() StorageType {
	return TypeFloat
}
