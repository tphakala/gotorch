package storage

import (
	"archive/zip"
	"encoding/binary"
	"fmt"
	"sync"
)

type Int struct {
	base
	data []int32
}

var _ Storage = &Int{}

func (*Int) New(wg *sync.WaitGroup, size int, file *zip.File) (Storage, error) {
	fs, err := file.Open()
	if err != nil {
		return nil, fmt.Errorf("Int.New: can not open file %s: %w", file.Name, err)
	}
	defer func() { _ = fs.Close() }()
	var ret Int
	ret.data = make([]int32, size)
	wg.Go(func() {
		err = binary.Read(fs, binary.LittleEndian, ret.data)
		if err != nil {
			panic(fmt.Errorf("Int.New: can not read file %s: %w", file.Name, err))
		}
	})
	return &ret, nil
}

func (i *Int) Get() any {
	return i.data
}

func (*Int) Type() StorageType {
	return TypeInt
}
