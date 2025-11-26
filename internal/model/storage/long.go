package storage

import (
	"archive/zip"
	"encoding/binary"
	"fmt"
	"sync"
)

type Long struct {
	base
	data []int64
}

var _ Storage = &Long{}

func (*Long) New(wg *sync.WaitGroup, size int, file *zip.File) (Storage, error) {
	fs, err := file.Open()
	if err != nil {
		return nil, fmt.Errorf("Long.New: can not open file %s: %w", file.Name, err)
	}
	defer func() { _ = fs.Close() }()
	var ret Long
	ret.data = make([]int64, size)
	wg.Go(func() {
		err = binary.Read(fs, binary.LittleEndian, ret.data)
		if err != nil {
			panic(fmt.Errorf("Long.New: can not read file %s: %w", file.Name, err))
		}
	})
	return &ret, nil
}

func (l *Long) Get() any {
	return l.data
}

func (*Long) Type() StorageType {
	return TypeLong
}
