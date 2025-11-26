package storage

import (
	"archive/zip"
	"encoding/binary"
	"fmt"
	"sync"
)

type BFloat16 struct {
	base
	data []uint16
}

var _ Storage = &BFloat16{}

func (*BFloat16) New(wg *sync.WaitGroup, size int, file *zip.File) (Storage, error) {
	fs, err := file.Open()
	if err != nil {
		return nil, fmt.Errorf("BFloat16.New: can not open file %s: %w", file.Name, err)
	}
	defer func() { _ = fs.Close() }()
	var ret BFloat16
	ret.data = make([]uint16, size)
	wg.Go(func() {
		err = binary.Read(fs, binary.LittleEndian, ret.data)
		if err != nil {
			panic(fmt.Errorf("BFloat16.New: can not read file %s: %w", file.Name, err))
		}
	})
	return &ret, nil
}

func (f *BFloat16) Get() any {
	return f.data
}

func (*BFloat16) Type() StorageType {
	return TypeBFloat16
}
