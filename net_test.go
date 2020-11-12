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
func TestMLPForwardBackwardStep(t *testing.T) {
	neuron.Logf(1, "Running TestNewMLP\n")

	// Seed rand so we get the same weights.
	rand.Seed(12)

	arch := []int{2, 3, 2, 1}
	n := neuron.NewMLP(arch)

	n.Start(true, 1, 1.0)

	const outWant = 8.4846442116e-05
	output := n.Forward([]float64{1.123, -2.234})
	neuron.Logf(1, "Output: %v\n", output)
	if !almostEqual(output[0], outWant) {
		t.Errorf("MLP output is %.10e; expected %.4e", output[0], outWant)
	}

	const gradWant = 2.5042884999258233e-07
	gradData := n.Backward([]float64{1.0})
	neuron.Logf(1, "Grad wrt data: %v\n", gradData)
	if !almostEqual(gradData[0], gradWant) {
		t.Errorf("MLP grad output is %.10e; expected %.4e", gradData[0], gradWant)
	}
}
