// Package neuron implements multi-layer neural networks with concurrent
// neurons.
//
// Individual neurons execute independently using goroutines and communicate via
// channels--both forwards and backwards. Networks consist of multiple
// fully-connected layers of neurons. SGD is used for training.
package neuron

import (
	"fmt"
	"math/rand"
)

// A Unit is a single neuron unit with weights, a bias, and input/output
// channels for forward and backward. Weights are represented as maps from
// string unit IDs to values.
type Unit struct {
	ID    string
	W     *Weight
	nin   int
	activ Activation
	opt   Optimizer
	// Single input channel.
	input chan signal
	// output channels for each downstream connection.
	output map[string](chan signal)
	// Similarly, input and output channels for backwards communication.
	inputB  chan signal
	outputB map[string](chan signal)
	// Channel to keep track of when the update is done.
	stepDone chan int
}

// A Weight represents a neuron's weight map.
type Weight struct {
	Params map[string]*Param
}

func (w *Weight) init(id string, data float64, requiresGrad bool) {
	w.Params[id] = &Param{
		Data:         data,
		RequiresGrad: requiresGrad,
	}
}

func (w *Weight) forward(id string, value float64) float64 {
	p, ok := w.Params[id]
	if !ok {
		return 0.0
	}
	if p.RequiresGrad {
		p.value = value
	}
	return p.Data * value
}

func (w *Weight) backward(id string, grad float64) float64 {
	p, ok := w.Params[id]
	if !ok {
		return 0.0
	}
	if p.RequiresGrad {
		p.grad += grad * p.value
	}
	return p.Data * grad
}

// NewWeight creates a new weight map.
func NewWeight() *Weight {
	w := Weight{
		Params: make(map[string]*Param),
	}
	return &w
}

// A Param is a neural network parameter
type Param struct {
	Data         float64
	RequiresGrad bool
	value        float64
	grad         float64
}

// signals are used to communicate between neuron Units.
type signal struct {
	id    string
	value float64
}

// special IDs for input and output channels and bias parameters.
const (
	inputID  = "_INPUT"
	outputID = "_OUTPUT"
	biasID   = "_BIAS"
)

func newInputUnit(id string, stepDone chan int) *Unit {
	activ := new(Identity)
	u := newUnit(id, activ, stepDone)
	u.feedIn()
	return u
}

func newHiddenUnit(id string, stepDone chan int) *Unit {
	activ := new(Relu)
	u := newUnit(id, activ, stepDone)
	u.W.init(biasID, 0.1, true)
	return u
}

func newOutputUnit(id string, stepDone chan int) *Unit {
	activ := new(Identity)
	u := newUnit(id, activ, stepDone)
	u.W.init(biasID, 0.0, true)
	u.feedOut()
	return u
}

// Create a new Unit with a given string id and layer type.
func newUnit(id string, activ Activation, stepDone chan int) *Unit {
	u := Unit{
		ID:       id,
		W:        NewWeight(),
		activ:    activ,
		input:    make(chan signal),
		output:   make(map[string](chan signal)),
		inputB:   make(chan signal),
		outputB:  make(map[string](chan signal)),
		stepDone: stepDone,
	}

	logf(2, "New unit %s\n", id)
	return &u
}

// Connect two units together in series: u1 -> u2.
func (u *Unit) connect(u2 *Unit) {
	u.output[u2.ID] = u2.input
	u2.W.init(u.ID, randUnif(-0.01, 0.01), true)
	u2.outputB[u.ID] = u.inputB
	u2.nin++
	logf(2, "Connect: %s -> %s\n", u.ID, u2.ID)
}

// Create an input connection to a unit.
func (u *Unit) feedIn() {
	u.W.init(inputID, 1.0, false)
	u.nin++
}

// Create an output channel from a unit.
func (u *Unit) feedOut() {
	u.output[outputID] = make(chan signal)
}

// Initialize a weight value by sampling randomly from [-0.01, 0.01).
func randUnif(a, b float64) float64 {
	w := rand.Float64()
	w = a + (b-a)*w
	return w
}

func (u *Unit) setOptimizer(opt Optimizer) {
	u.opt = opt.New()
}

// Forward pass through the unit. Collects input from all incoming units and
// fires an activation.
func (u *Unit) forward() {
	var s signal
	// Accumulate weighted inputs from input connections.
	// NOTE: assuming only one received activation per input unit.
	act := u.W.forward(biasID, 1.0)
	for ii := 0; ii < u.nin; ii++ {
		s = <-u.input
		act += u.W.forward(s.id, s.value)
	}

	// Fire activation
	act = u.activ.Forward(act)
	s = signal{id: u.ID, value: act}
	for k := range u.output {
		u.output[k] <- s
	}
}

// Backward pass through the unit. Waits for gradients from all downstream
// connections, updates weight gradients, and back-propagates.
func (u *Unit) backward() {
	var s signal
	// Accumulate grads from all output connections.
	grad := 0.0
	for ii := 0; ii < len(u.output); ii++ {
		s = <-u.inputB
		grad += s.value
	}

	// Backprop.
	grad = u.activ.Backward(grad)
	for k := range u.W.Params {
		gradi := u.W.backward(k, grad)
		if c, ok := u.outputB[k]; ok {
			c <- signal{id: u.ID, value: gradi}
		}
	}
}

// Update the weights and bias by taking a gradient descent step.
func (u *Unit) step() {
	if u.opt == nil {
		panic(fmt.Sprintf("Unit %s optimizer is uninitialized!", u.ID))
	}
	for k, p := range u.W.Params {
		u.opt.Step(k, p)
	}
}

// Start starts an endless loop of forward and backward passes with periodic
// gradient updates.
func (u *Unit) start(train bool, updateFreq int) {
	step := 1
	for {
		u.forward()
		if train {
			u.backward()
			if updateFreq > 0 && step%updateFreq == 0 {
				u.step()
			}
		}
		step++
		u.stepDone <- 1
	}
}
