# gotorch

[![gotorch](https://github.com/lwch/gotorch/actions/workflows/cpu.yml/badge.svg)](https://github.com/lwch/gotorch/actions/workflows/cpu.yml)
[![gotorch](https://github.com/lwch/gotorch/actions/workflows/gpu.yml/badge.svg)](https://github.com/lwch/gotorch/actions/workflows/gpu.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/lwch/gotorch.svg)](https://pkg.go.dev/github.com/lwch/gotorch)

This is a Go wrapper library for libtorch. With this library, you can quickly build torch models. It currently supports the latest version of libtorch (2.0.1) and the following operating systems:

- Windows
- Linux
- macOS

Both *CPU* and *GPU* computation are supported.

## Installation

1. Download [libtorch](https://pytorch.org/get-started/locally/). On Windows, extract it to the D: drive. On Linux and macOS, extract it to the /usr/local/lib directory.
2. Download [libgotorch](https://github.com/lwch/gotorch/releases/latest) and place it in the libtorch lib directory:
   - On Windows, rename it to gotorch.dll
   - On Linux, download the appropriate .so file based on your glibc version and rename it to libgotorch.so
   - On macOS, the latest version only supports arm64 architecture. After downloading, rename it to libgotorch.dylib

Note: Since the official Windows version of libtorch is compiled with MSVC and cannot be linked properly with MinGW, the libgotorch library was added for conversion. For information on compiling libgotorch, see [libgotorch compilation](docs/libgotorch.md). You can also refer to the commands in [release.yml](.github/workflows/release.yml).

### Linux

Add the following to your .bashrc:

```
export LIBRARY_PATH="$LIBRARY_PATH:/usr/local/lib/libtorch/lib"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib/libtorch/lib"
```

### macOS

Add the following to your .bashrc:

```
export LIBRARY_PATH="$LIBRARY_PATH:/usr/local/lib/libtorch/lib"
export DYLD_FALLBACK_LIBRARY_PATH="$DYLD_FALLBACK_LIBRARY_PATH:/usr/local/lib/libtorch/lib"
```

### Windows

Using cgo on Windows requires MinGW. [llvm-mingw](https://github.com/mstorsjo/llvm-mingw) is recommended. Add the following environment variables:

```
LIBRARY_PATH="D:\libtorch\lib"
Path="D:\libtorch\lib;<path to mingw>\bin"
```

## Usage

See the example in [mlp](example/mlp):

```go
a := tensor.ARange(nil, 6, consts.KFloat,
    tensor.WithShape(2, 3),
    tensor.WithDevice(consts.KCUDA))
b := tensor.ARange(nil, 6, consts.KFloat,
    tensor.WithShape(3, 2),
    tensor.WithDevice(consts.KCUDA))
c := a.MatMul(b)
fmt.Println(c.ToDevice(consts.KCPU).Float32Value()) // Note: Data in GPU memory cannot be read directly; it must be transferred to CPU first
```

**Note: Since most tensor objects are created on the C stack and Go cannot accurately track memory usage, it is recommended to disable Go's GC using debug.SetGCPercent in long-running services (such as model training) and manually call runtime.GC in each iteration to release memory.**

## Loading Model Checkpoints

```go
m, _ := model.Load("yolo_tiny.pt", nil)
for name, t := m.Params() {
    fmt.Println(name, t.Shapes())
}
```

## Version Compatibility

| gotorch version | libtorch version |
| --- | --- |
| v1.0.0~v1.5.7 | v2.0.1 |
| v1.6.0~v1.7.2 | v2.1~v2.2.1 |
| v1.7.3 | v2.2.2 |
| v1.7.4~v1.8.0 | v2.3.1 |
| v1.9.0~v1.9.2 | v2.4.x |
| v1.9.3 | v2.5.x |
| v1.9.4 | v2.6.x |
| v1.9.5 | v2.7.x |
| v1.9.11 | v2.8.x |
| v1.9.12 | v2.9.x |
