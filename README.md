# go-neuron

Multi-layer neural networks with concurrent neurons.

[![Go Report Card](https://goreportcard.com/badge/github.com/clane9/go-neuron)](https://goreportcard.com/report/github.com/clane9/go-neuron)
[![GoDoc](https://godoc.org/github.com/clane9/go-neuron?status.svg)](https://godoc.org/github.com/clane9/go-neuron)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Individual neurons execute independently using goroutines and communicate via
channels--both forwards and backwards. Networks consist of multiple
fully-connected layers of neurons. SGD is used for training.

## Installation

```
go get github.com/clane9/go-neuron
```

## Example

[`gaussian_data.go`](examples/gaussian_data.go) trains a simple MLP to
classify two-class Gaussian data. Here is the `main` function.

```golang
func main() {
  // MLP with a 64-dim input, scalar output, and two 128-dim hidden layers.
  arch := []int{64, 128, 128, 1}
  n := neuron.NewMLP(arch)
  // Start the network for training with the given "batch size" and learning rate.
  n.Start(true, 32, 1.0e-03)

  // Train for 1000 steps.
  for ii := 0; ii <= 1000; ii++ {
    data, target = gaussianData(64)
    score = n.Forward(data)
    loss, grad = neuron.MarginLoss(score[0], target)
    // Weights are updated automatically after backward.
    n.Backward([]float64{grad})
  }
}
```

## Performance

By representing neurons as independent computational units, we can naturally take advantage of multi-core CPUs. For example, `gaussian_data.go` achieves near linear speedup with increasing `GOMAXPROCS` on a quad-core MBP.

When compared to standard deep learning frameworks however, the implementation is pretty slow. ([`BenchmarkMLP`](net_test.go) runs ~10x faster in `pytorch` on the same hardware.) This is not all that surprising. Matrix multiplication is very efficient after all.

## Similar alternatives

- [Varis](https://github.com/Xamber/Varis)
- [gonn](https://github.com/fxsjy/gonn)
- [go-deep](https://github.com/patrikeh/go-deep)
- [gobrain](https://github.com/goml/gobrain)
- [golearn/neural](https://github.com/golang-basic/golearn/tree/master/neural)