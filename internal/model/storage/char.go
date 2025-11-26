package storage

import (
	"archive/zip"
	"encoding/binary"
	"fmt"
	"sync"
)

type Char struct {
	base
	data []int8
}

var _ Storage = &Char{}

func (*Char) New(wg *sync.WaitGroup, size int, file *zip.File) (Storage, error) {
	fs, err := file.Open()
	if err != nil {
		return nil, fmt.Errorf("Char.New: can not open file %s: %w", file.Name, err)
	}
	defer func() { _ = fs.Close() }()
	var ret Char
	ret.data = make([]int8, size)
	wg.Go(func() {
		err = binary.Read(fs, binary.LittleEndian, ret.data)
		if err != nil {
			panic(fmt.Errorf("Char.New: can not read file %s: %w", file.Name, err))
		}
	})
	return &ret, nil
}

func (c *Char) Get() any {
	return c.data
}

func (*Char) Type() StorageType {
	return TypeChar
}
