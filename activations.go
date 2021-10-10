package neuron

import (
	"math"
)

// An Activation represents a neural network activation function.
type Activation interface {
	Forward(float64) float64
	Backward(float64) float64
}

// Relu activation function.
type Relu struct {
	value float64
}

// Forward Relu activation
func (a *Relu) Forward(value float64) float64 {
	a.value = value
	return math.Max(value, 0)
}

// Backward pass of Relu gradient
func (a *Relu) Backward(grad float64) float64 {
	if a.value <= 0 {
		grad = 0.0
	}
	return grad
}

// Identity activation function
type Identity struct{}

// Forward Identity activation
func (a *Identity) Forward(value float64) float64 {
	return value
}

// Backward pass of Identity gradient
func (a *Identity) Backward(grad float64) float64 {
	return grad
}
