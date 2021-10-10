package neuron

import (
	"testing"
)

// Test ReLU
func TestReluActivation(t *testing.T) {
	relu := new(Relu)

	x := 1.0
	z := relu.Forward(x)
	g := relu.Backward(1.0)
	if z != 1.0 || g != 1.0 {
		t.Errorf("Invalid Relu")
	}

	x = -1.0
	z = relu.Forward(x)
	g = relu.Backward(1.0)
	if z != 0.0 || g != 0.0 {
		t.Errorf("Invalid Relu")
	}
}
