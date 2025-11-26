package storage

import (
	"archive/zip"
	"encoding/binary"
	"fmt"
	"sync"
)

type Short struct {
	base
	data []int16
}

var _ Storage = &Short{}

func (*Short) New(wg *sync.WaitGroup, size int, file *zip.File) (Storage, error) {
	fs, err := file.Open()
	if err != nil {
		return nil, fmt.Errorf("Short.New: can not open file %s: %w", file.Name, err)
	}
	defer func() { _ = fs.Close() }()
	var ret Short
	ret.data = make([]int16, size)
	wg.Go(func() {
		err = binary.Read(fs, binary.LittleEndian, ret.data)
		if err != nil {
			panic(fmt.Errorf("Short.New: can not read file %s: %w", file.Name, err))
		}
	})
	return &ret, nil
}

func (s *Short) Get() any {
	return s.data
}

func (*Short) Type() StorageType {
	return TypeShort
}
