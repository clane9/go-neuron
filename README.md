# go-neuron

Multi-layer neural networks with concurrent neurons.

[![GoDoc](https://godoc.org/github.com/clane9/go-neuron?status.svg)](https://godoc.org/github.com/clane9/go-neuron)
[![Build Status](https://travis-ci.com/clane9/go-neuron.svg?branch=main)](https://travis-ci.com/clane9/go-neuron?branch=main)
[![Go Report Card](https://goreportcard.com/badge/github.com/clane9/go-neuron)](https://goreportcard.com/report/github.com/clane9/go-neuron)
[![codecov](https://codecov.io/gh/clane9/go-neuron/branch/main/graph/badge.svg)](https://codecov.io/gh/clane9/go-neuron)
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
  opt := neuron.NewSGD(1.0e-01, 0.9, 1.0e-05)
  n := neuron.NewMLP(arch, opt)
  // Start the network for training with the given "batch size"
  n.Start(true, 32)

  // Train for 200 steps.
  for ii := 0; ii <= 200; ii++ {
    data, target = gaussianData(64)
    score = n.Forward(data)
    loss, grad = neuron.MarginLoss(score[0], target)
    // Weights are updated automatically after backward.
    n.Backward([]float64{grad})
  }
}
```

## Performance

Because our neurons run concurrently, we can achieve some speedup with
multi-core CPUs. When compared to standard deep learning frameworks however, the
implementation is pretty slow. ([`BenchmarkMLP`](net_test.go) runs ~10x faster
in `pytorch` on the same hardware.)

## Similar projects

- [Varis](https://github.com/Xamber/Varis)
- [gonn](https://github.com/fxsjy/gonn)
- [go-deep](https://github.com/patrikeh/go-deep)
- [gobrain](https://github.com/goml/gobrain)
- [golearn/neural](https://github.com/golang-basic/golearn/tree/master/neural)