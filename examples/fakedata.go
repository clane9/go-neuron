package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/clane9/go-neuron"
)

func main() {
	rand.Seed(2020)

	const steps = 10000
	const inDim = 64
	const outDim = 1
	neuron.Verbosity = 0

	arch := []int{inDim, 128, 128, outDim}
	n := neuron.NewMLP(arch)
	n.Start(true, 32, 1.0e-05)

	var data []float64
	var score []float64
	var target int
	var loss float64
	var grad float64
	for ii := 1; ii <= steps; ii++ {
		data, target = genFakeData(inDim)
		score = n.Forward(data)
		loss, grad = neuron.MarginLoss(score[0], target)
		n.Backward([]float64{grad})
		n.Sync()

		if ii%10 == 0 {
			t := time.Now()
			fmt.Printf("(%s) step=%06d loss=%.5e gradL=%.5e\n",
				t.Format("15:04:05.999"), ii, loss, grad)
		}
	}
}

// Generate a fake data vector by sampling uniformly from [-1, 1]
func genFakeData(n int) (data []float64, target int) {
	data = make([]float64, n)
	for ii := 0; ii < n; ii++ {
		data[ii] = 2.0*rand.Float64() - 1.0
	}
	target = 2*rand.Intn(2) - 1
	return
}
