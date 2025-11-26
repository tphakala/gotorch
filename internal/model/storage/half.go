package storage

import (
	"archive/zip"
	"encoding/binary"
	"fmt"
	"sync"
)

type Half struct {
	base
	data []uint16
}

var _ Storage = &Half{}

func (*Half) New(wg *sync.WaitGroup, size int, file *zip.File) (Storage, error) {
	fs, err := file.Open()
	if err != nil {
		return nil, fmt.Errorf("Half.New: can not open file %s: %w", file.Name, err)
	}
	defer func() { _ = fs.Close() }()
	var ret Half
	ret.data = make([]uint16, size)
	wg.Go(func() {
		err = binary.Read(fs, binary.LittleEndian, ret.data)
		if err != nil {
			panic(fmt.Errorf("Half.New: can not read file %s: %w", file.Name, err))
		}
	})
	return &ret, nil
}

func (f *Half) Get() any {
	return f.data
}

func (*Half) Type() StorageType {
	return TypeHalf
}
