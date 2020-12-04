// Train an MLP to classify two-class Gaussian data.

package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/clane9/go-neuron"
)

func main() {
	rand.Seed(2020)

	const (
		steps  = 3000
		inDim  = 64
		outDim = 1
	)
	neuron.Verbosity = 0

	// MLP with two 128-dim hidden layers.
	arch := []int{inDim, 128, 128, outDim}
	n := neuron.NewMLP(arch)
	// Start the network running for training. Gradients accumulate for 32 inputs
	// before updating. (This is equivalent to mini-batch gradient descent.)
	n.Start(true, 32, 1.0e-03)

	var (
		data   []float64
		score  []float64
		target int
		loss   float64
		grad   float64
	)

	// Training loop
	start := time.Now()
	for ii := 1; ii <= steps; ii++ {
		data, target = gaussianData(inDim)
		score = n.Forward(data)
		loss, grad = neuron.MarginLoss(score[0], target)
		n.Backward([]float64{grad})

		if ii%50 == 0 {
			t := time.Now()
			fmt.Printf("(%s)\tstep=%06d\tloss=%.5e\tgradL=%.5e\n",
				t.Format("15:04:05.999"), ii, loss, grad)
		}
	}
	elapsed := time.Since(start)
	fmt.Printf("Done %d steps in %.2fs (%.2f steps/s)\n",
		steps, elapsed.Seconds(), float64(steps)/elapsed.Seconds())
}

// Generate a random data sample drawn from a two class Gaussian mixture.
func gaussianData(n int) (data []float64, target int) {
	target = 2*rand.Intn(2) - 1
	data = make([]float64, n)
	for ii := 0; ii < n; ii++ {
		data[ii] = rand.NormFloat64() + 2.0*float64(target)
	}
	return
}
