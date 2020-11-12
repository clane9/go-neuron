package neuron_test

import (
	"math/rand"
	"testing"

	"github.com/clane9/go-neuron"
)

// Test construction of a new MLP network
func TestNewMLP(t *testing.T) {
	neuron.Logf(1, "Running TestNewMLP\n")

	arch := []int{2, 4, 4, 1}
	n := neuron.NewMLP(arch)
	for ii, sz := range arch {
		if n.Arch[ii] != sz {
			t.Errorf("Layer %d size is %d; expected %d", ii, n.Arch[ii], sz)
		}
	}

	// Check that invalid architectures are checked.
	arch = []int{2, 4}
	assertPanic(t, func() { neuron.NewMLP(arch) })
	arch = []int{2, 4, -1}
	assertPanic(t, func() { neuron.NewMLP(arch) })
}

// Test full forward/backward/step loop for the entire MLP.
func TestMLP(t *testing.T) {
	neuron.Logf(1, "Running TestMLP\n")

	// Seed rand so we get the same weights.
	rand.Seed(12)

	arch := []int{2, 3, 2, 1}
	n := neuron.NewMLP(arch)

	n.Start(true, 1, 1.0)
	output := n.Forward([]float64{1.123, -2.234})
	n.Backward([]float64{1.0})
	n.Sync()

	const outWant = 8.4846442116e-05
	neuron.Logf(1, "Output: %v\n", output)
	if !almostEqual(output[0], outWant) {
		t.Errorf("MLP output is %.10e; expected %.4e", output[0], outWant)
	}
}

func BenchmarkMLP(b *testing.B) {
	neuron.Verbosity = 0

	// Seed rand so we get the same weights.
	rand.Seed(12)

	const inDim = 64
	const outDim = 1
	arch := []int{inDim, 128, 128, outDim}
	n := neuron.NewMLP(arch)

	input := make([]float64, inDim)
	for ii := 0; ii < inDim; ii++ {
		input[ii] = rand.Float64()
	}
	grad := []float64{1.0}

	// lr set to 0 so we don't acually update the weights
	n.Start(true, 1, 0.0)

	for ii := 0; ii < b.N; ii++ {
		n.Forward(input)
		n.Backward(grad)
		n.Sync()
	}
}
