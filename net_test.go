package neuron_test

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/clane9/go-neuron"
)

// Test construction of a new MLP network
func TestNewMLP(t *testing.T) {
	fmt.Printf("Running TestNewMLP\n")

	arch := []int{2, 4, 4, 1}
	opt := neuron.NewSGD(1.0, 0.0, 0.0)
	n := neuron.NewMLP(arch, opt)
	for ii, sz := range arch {
		if n.Arch[ii] != sz {
			t.Errorf("Layer %d size is %d; expected %d", ii, n.Arch[ii], sz)
		}
	}

	// Check that invalid architectures are checked.
	arch = []int{2, 4}
	assertPanic(t, func() { neuron.NewMLP(arch, opt) })
	arch = []int{2, 4, -1}
	assertPanic(t, func() { neuron.NewMLP(arch, opt) })
}

// Test full forward/backward/step loop for the entire MLP.
func TestMLP(t *testing.T) {
	fmt.Printf("Running TestMLP\n")

	// Seed rand so we get the same weights.
	rand.Seed(12)

	arch := []int{2, 3, 2, 1}
	opt := neuron.NewSGD(1.0, 0.0, 0.0)
	n := neuron.NewMLP(arch, opt)

	n.Start(true, 1)
	output := n.Forward([]float64{1.123, -2.234})
	n.Backward([]float64{1.0})

	const outWant = 8.4846442116e-05
	fmt.Printf("Output: %v\n", output)
	if !almostEqual(output[0], outWant) {
		t.Errorf("MLP output is %.10e; expected %.4e", output[0], outWant)
	}
}

// Benchmark a full forward/backward/step loop.
// This is pretty slow! 3.4 ms per op, compared to 0.4 ms in pytorch (using the
// same architecture and machine, cpu only). Not all that surprising, matrix
// multiplication is very efficient after all.
func BenchmarkMLP(b *testing.B) {
	neuron.Verbosity = 0

	// Seed rand so we get the same weights.
	rand.Seed(12)

	const inDim = 64
	const outDim = 1
	arch := []int{inDim, 128, 128, outDim}
	// lr set to 0 so we don't acually update the weights
	opt := neuron.NewSGD(0.0, 0.0, 0.0)
	n := neuron.NewMLP(arch, opt)

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
