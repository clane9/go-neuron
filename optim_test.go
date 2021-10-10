package neuron

import (
	"testing"
)

// Test margin loss.
func TestSGD(t *testing.T) {
	const id = "000"
	p := &Param{
		Data:         1.0,
		RequiresGrad: true,
		grad:         1.0,
	}
	opt := NewSGD(0.1, 0.0, 1.0)

	opt.Step(id, p)
	if !almostEqual(p.Data, 0.8) {
		t.Errorf("Incorrect SGD step")
	}

	p.grad = 1.0
	opt.Momentum = 0.9
	opt.WeightDecay = 0.0
	opt.Step(id, p)
	if !almostEqual(p.Data, 0.7) {
		t.Errorf("Incorrect SGD step")
	}

	// 0.9 * 1.0 + -1.0 = -0.1
	p.grad = -1.0
	opt.Step(id, p)
	if !almostEqual(opt.buf[id], -0.1) {
		t.Errorf("Incorrect SGD step")
	}
	if !almostEqual(p.Data, 0.71) {
		t.Errorf("Incorrect SGD step")
	}
}
