package jit

import (
	"os"
	"path/filepath"
	"sync"
	"testing"
	"testing/synctest"

	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
)

// testModelPath returns the path to the BirdNET test model.
// Set BIRDNET_MODEL_PATH env var or use default location.
func testModelPath() string {
	if p := os.Getenv("BIRDNET_MODEL_PATH"); p != "" {
		return p
	}
	return "/home/thakala/BirdNET+_V3.0-preview2_EUNA_1K_FP32.pt"
}

// skipIfNoModel skips the test if the model file doesn't exist.
func skipIfNoModel(t *testing.T) string {
	t.Helper()
	path := testModelPath()
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("model file not found: %s (set BIRDNET_MODEL_PATH to override)", path)
	}
	return path
}

func TestLoad(t *testing.T) {
	path := skipIfNoModel(t)

	model, err := Load(path)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	// Model should be in eval mode by default
	model.Eval()
}

func TestLoadToDevice(t *testing.T) {
	path := skipIfNoModel(t)

	// Test CPU loading
	t.Run("CPU", func(t *testing.T) {
		model, err := LoadToDevice(path, consts.KCPU)
		if err != nil {
			t.Fatalf("failed to load model to CPU: %v", err)
		}
		defer model.Close()
	})

	// Test CUDA loading (skip if not available)
	t.Run("CUDA", func(t *testing.T) {
		model, err := LoadToDevice(path, consts.KCUDA)
		if err != nil {
			t.Skipf("CUDA not available: %v", err)
		}
		defer model.Close()
	})
}

func TestLoadError(t *testing.T) {
	_, err := Load("/nonexistent/path/to/model.pt")
	if err == nil {
		t.Fatal("expected error for nonexistent model")
	}
}

func TestForward(t *testing.T) {
	path := skipIfNoModel(t)

	model, err := Load(path)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	// BirdNET expects input shape [batch, 160000] (5s @ 32kHz)
	input := tensor.FromFloat32(make([]float32, 160000),
		tensor.WithShapes(1, 160000),
		tensor.WithDevice(consts.KCPU))

	output, err := model.Forward(input)
	if err != nil {
		t.Fatalf("forward pass failed: %v", err)
	}

	// Should return first output (embeddings [1, 1280])
	shapes := output.Shapes()
	t.Logf("Forward output shape: %v", shapes)

	if len(shapes) != 2 {
		t.Errorf("expected 2D output, got %d dimensions", len(shapes))
	}
	if shapes[0] != 1 {
		t.Errorf("expected batch size 1, got %d", shapes[0])
	}
}

func TestForwardMulti(t *testing.T) {
	path := skipIfNoModel(t)

	model, err := Load(path)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	// BirdNET expects input shape [batch, 160000] (5s @ 32kHz)
	input := tensor.FromFloat32(make([]float32, 160000),
		tensor.WithShapes(1, 160000),
		tensor.WithDevice(consts.KCPU))

	outputs, err := model.ForwardMulti(input, 2)
	if err != nil {
		t.Fatalf("forward pass failed: %v", err)
	}

	if len(outputs) != 2 {
		t.Fatalf("expected 2 outputs, got %d", len(outputs))
	}

	// Output 0: embeddings [1, 1280]
	embeddingShape := outputs[0].Shapes()
	t.Logf("Embeddings shape: %v", embeddingShape)
	if len(embeddingShape) != 2 || embeddingShape[1] != 1280 {
		t.Errorf("expected embeddings shape [1, 1280], got %v", embeddingShape)
	}

	// Output 1: predictions [1, 1225] (species logits)
	predictionShape := outputs[1].Shapes()
	t.Logf("Predictions shape: %v", predictionShape)
	if len(predictionShape) != 2 || predictionShape[1] != 1225 {
		t.Errorf("expected predictions shape [1, 1225], got %v", predictionShape)
	}
}

func TestOutputsIterator(t *testing.T) {
	path := skipIfNoModel(t)

	model, err := Load(path)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	input := tensor.FromFloat32(make([]float32, 160000),
		tensor.WithShapes(1, 160000),
		tensor.WithDevice(consts.KCPU))

	outputs, err := model.ForwardMulti(input, 2)
	if err != nil {
		t.Fatalf("forward pass failed: %v", err)
	}

	// Test the Outputs iterator (Go 1.23+ iter.Seq2)
	count := 0
	for idx, out := range Outputs(outputs) {
		t.Logf("Output %d shape: %v", idx, out.Shapes())
		count++
	}

	if count != 2 {
		t.Errorf("expected 2 iterations, got %d", count)
	}
}

func TestEvalTrainModes(t *testing.T) {
	path := skipIfNoModel(t)

	model, err := Load(path)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	// Should not panic
	model.Eval()
	model.Train()
	model.Eval()
}

func TestClose(t *testing.T) {
	path := skipIfNoModel(t)

	model, err := Load(path)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}

	// Close should be idempotent
	model.Close()
	model.Close()
}

func TestConcurrentInference(t *testing.T) {
	path := skipIfNoModel(t)

	// Use Go 1.25 synctest for deterministic concurrent testing
	synctest.Test(t, func(t *testing.T) {
		model, err := Load(path)
		if err != nil {
			t.Fatalf("failed to load model: %v", err)
		}
		defer model.Close()

		const numGoroutines = 4
		var wg sync.WaitGroup
		errors := make(chan error, numGoroutines)

		for range numGoroutines {
			wg.Add(1)
			go func() {
				defer wg.Done()

				input := tensor.FromFloat32(make([]float32, 160000),
					tensor.WithShapes(1, 160000),
					tensor.WithDevice(consts.KCPU))

				_, err := model.Forward(input)
				if err != nil {
					errors <- err
				}
			}()
		}

		wg.Wait()
		close(errors)

		for err := range errors {
			t.Errorf("concurrent inference failed: %v", err)
		}
	})
}

func TestBatchInference(t *testing.T) {
	path := skipIfNoModel(t)

	model, err := Load(path)
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	// Test with batch size 4
	batchSize := int64(4)
	input := tensor.FromFloat32(make([]float32, int(batchSize)*160000),
		tensor.WithShapes(batchSize, 160000),
		tensor.WithDevice(consts.KCPU))

	outputs, err := model.ForwardMulti(input, 2)
	if err != nil {
		t.Fatalf("batch forward pass failed: %v", err)
	}

	// Check batch dimension is preserved
	embeddingShape := outputs[0].Shapes()
	if embeddingShape[0] != batchSize {
		t.Errorf("expected batch size %d in embeddings, got %d", batchSize, embeddingShape[0])
	}

	predictionShape := outputs[1].Shapes()
	if predictionShape[0] != batchSize {
		t.Errorf("expected batch size %d in predictions, got %d", batchSize, predictionShape[0])
	}

	t.Logf("Batch embeddings shape: %v", embeddingShape)
	t.Logf("Batch predictions shape: %v", predictionShape)
}

func BenchmarkForward(b *testing.B) {
	path := testModelPath()
	if _, err := os.Stat(path); os.IsNotExist(err) {
		b.Skipf("model file not found: %s", path)
	}

	model, err := Load(path)
	if err != nil {
		b.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	input := tensor.FromFloat32(make([]float32, 160000),
		tensor.WithShapes(1, 160000),
		tensor.WithDevice(consts.KCPU))

	b.ResetTimer()
	for b.Loop() {
		_, err := model.Forward(input)
		if err != nil {
			b.Fatalf("forward pass failed: %v", err)
		}
	}
}

func BenchmarkForwardMulti(b *testing.B) {
	path := testModelPath()
	if _, err := os.Stat(path); os.IsNotExist(err) {
		b.Skipf("model file not found: %s", path)
	}

	model, err := Load(path)
	if err != nil {
		b.Fatalf("failed to load model: %v", err)
	}
	defer model.Close()

	input := tensor.FromFloat32(make([]float32, 160000),
		tensor.WithShapes(1, 160000),
		tensor.WithDevice(consts.KCPU))

	b.ResetTimer()
	for b.Loop() {
		_, err := model.ForwardMulti(input, 2)
		if err != nil {
			b.Fatalf("forward pass failed: %v", err)
		}
	}
}

func TestModelPath(t *testing.T) {
	path := testModelPath()
	absPath, err := filepath.Abs(path)
	if err != nil {
		t.Fatalf("failed to get absolute path: %v", err)
	}
	t.Logf("Test model path: %s", absPath)

	info, err := os.Stat(path)
	if err != nil {
		t.Skipf("model not found: %v", err)
	}
	t.Logf("Model size: %.2f MB", float64(info.Size())/(1024*1024))
}
