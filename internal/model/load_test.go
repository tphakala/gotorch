package model

import (
	"fmt"
	"testing"
)

func TestLoad(t *testing.T) {
	m, err := Load("./test/linear.pt")
	if err != nil {
		t.Fatal(err)
	}
	for i := range 1000 {
		fmt.Println(m.params[fmt.Sprintf("linear.%d.linear", i)].Get())
	}
}
