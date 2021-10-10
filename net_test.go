package neuron

import (
	"fmt"
	"math/rand"
	"testing"
)

// Test construction of a new MLP network
func TestNewMLP(t *testing.T) {
	fmt.Printf("Running TestNewMLP\n")

	arch := []int{2, 4, 4, 1}
	opt := NewSGD(1.0, 0.0, 0.0)
	n := NewMLP(arch, opt)
	for ii, sz := range arch {
		if n.Arch[ii] != sz {
			t.Errorf("Layer %d size is %d; expected %d", ii, n.Arch[ii], sz)
		}
	}

	// Check that invalid architectures are checked.
	arch = []int{2, 4}
	assertPanic(t, func() { NewMLP(arch, opt) })
	arch = []int{2, 4, -1}
	assertPanic(t, func() { NewMLP(arch, opt) })
}

// Test full forward/backward/step loop for the entire MLP.
func TestMLP(t *testing.T) {
	fmt.Printf("Running TestMLP\n")

	// Seed rand so we get the same weights.
	rand.Seed(12)

	arch := []int{2, 3, 2, 1}
	opt := NewSGD(1.0, 0.9, 1.0e-04)
	n := NewMLP(arch, opt)

	n.Start(true, 1)
	output := n.Forward([]float64{1.123, -2.234})
	n.Backward([]float64{1.0})

	const outWant = 8.4846442116e-05
	if !almostEqual(output[0], outWant) {
		t.Errorf("MLP output is %.10e; expected %.4e", output[0], outWant)
	}

	const weightWant = -1.0043559969e-01
	const id = "002_000000"
	weight := n.Layers[3][0].W.Params[id].Data
	if !almostEqual(weight, weightWant) {
		t.Errorf("Weight %s -> 003_000000 is %.10e; expected %.4e", id, weight, weightWant)
	}

	// Check that invalid args are checked.
	assertPanic(t, func() { n.Forward([]float64{1.123}) })
	// TODO: If backward is called without a forward it blocks because each unit
	// is blocked on forward. This should be tracked and handled inside net?
	assertPanic(t, func() { n.Backward([]float64{1.123, -2.234}) })
}

// Benchmark a full forward/backward/step loop.
// This is pretty slow! 3.4 ms per op, compared to 0.4 ms in pytorch (using the
// same architecture and machine, cpu only). Not all that surprising, matrix
// multiplication is very efficient after all.
func BenchmarkMLP(b *testing.B) {
	Verbosity = 0

	// Seed rand so we get the same weights.
	rand.Seed(12)

	const inDim = 64
	const outDim = 1
	arch := []int{inDim, 128, 128, outDim}
	// lr set to 0 so we don't acually update the weights
	opt := NewSGD(0.0, 0.0, 0.0)
	n := NewMLP(arch, opt)

	input := make([]float64, inDim)
	for ii := 0; ii < inDim; ii++ {
		input[ii] = rand.Float64()
	}
	grad := []float64{1.0}

	n.Start(true, 1)

	for ii := 0; ii < b.N; ii++ {
		n.Forward(input)
		n.Backward(grad)
	}
}
