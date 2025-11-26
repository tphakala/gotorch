package storage

import (
	"archive/zip"
	"fmt"
	"io"
	"sync"
)

type Byte struct {
	base
	data []byte
}

var _ Storage = &Byte{}

func (*Byte) New(wg *sync.WaitGroup, size int, file *zip.File) (Storage, error) {
	fs, err := file.Open()
	if err != nil {
		return nil, fmt.Errorf("Byte.New: can not open file %s: %w", file.Name, err)
	}
	defer func() { _ = fs.Close() }()
	var ret Byte
	ret.data = make([]byte, size)
	wg.Go(func() {
		_, err = io.ReadFull(fs, ret.data)
		if err != nil {
			panic(fmt.Errorf("Byte.New: can not read file %s: %w", file.Name, err))
		}
	})
	return &ret, nil
}

func (b *Byte) Get() any {
	return b.data
}

func (*Byte) Type() StorageType {
	return TypeByte
}
