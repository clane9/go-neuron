package neuron

import (
	"math"
	"math/rand"
)

// A Unit is a single neuron unit with weights input/output channels.
type Unit struct {
	id      string
	rectify bool
	preact  float64
	// Weights for each input connection.
	weight map[string]float64
	bias   float64
	// Values for each input connection.
	value map[string]float64
	// Single input channel.
	input chan signal
	// Output channels for each downstream connection.
	output map[string](chan signal)
	// Similarly, input and output channels for backwards communication.
	inputB  chan signal
	outputB map[string](chan signal)
}

type signal struct {
	id    string
	value float64
}

// NewUnit creates a new Unit with a given string id. It allocates new input
// channels and empty maps for weights, values, and outputs.
func NewUnit(id string, rectify bool) *Unit {
	u := Unit{
		id:      id,
		rectify: rectify,
		weight:  make(map[string]float64),
		bias:    0.1,
		value:   make(map[string]float64),
		// TODO: Need a large buffer to accommodate multiple units sending signals
		// simultaneously. But how big do I need?
		input:   make(chan signal, 512),
		output:  make(map[string](chan signal)),
		inputB:  make(chan signal, 512),
		outputB: make(map[string](chan signal)),
	}
	return &u
}

// Connect connects two units together in series: u1 -> u2.
func Connect(u1, u2 *Unit) {
	// Create forward connection from u1 -> u2 by giving u1 a reference to u2's
	// input channel.
	u1.output[u2.id] = u2.input
	// Initialize a weight value for u1 -> u2.
	u2.initWeight(u1.id)
	// Create backward connection from u1 <- u2 by giving u2 a reference to u1's
	// backward input channel.
	u2.outputB[u1.id] = u1.inputB
}

var rng = rand.New(rand.NewSource(12))

// Initialize a weight value by sampling randomly from [-0.01, 0.01).
func (u *Unit) initWeight(id string) float64 {
	w := rng.Float64()
	w = 0.02*w - 0.01
	u.weight[id] = w
	return w
}

// Zero out previous activation.
func (u *Unit) zero() {
	for k := range u.value {
		u.value[k] = 0.0
	}
	u.preact = u.bias
}

// Forward pass through the unit. Collects input from all incoming units and
// fires an activation.
// TODO: Sparser communication would be more efficient and "brain-like". I.e.,
// "fire" an activation only when non-zero. But it's not clear how to know when
// you've received all the inputs you're going to receive. Previously I used a
// timeout after first input. But this will miss inputs that are very delayed.
// This wouldn't be so bad, the delay would act like a kind of dropout. But I
// don't want stale inputs affecting the next forward pass.
func (u *Unit) forward() {
	u.zero()
	needRecv := make(map[string]bool)
	for k := range u.weight {
		needRecv[k] = true
	}
	for len(needRecv) > 0 {
		// Wait for a signal from one of the input connections.
		// Update the value and pre-activation.
		s := <-u.input
		u.value[s.id] += s.value
		u.preact += u.weight[s.id] * s.value
		delete(needRecv, s.id)
	}

	// Fire activation
	act := u.preact
	if u.rectify {
		act = math.Max(act, 0.0)
	}
	for k := range u.output {
		u.output[k] <- signal{id: u.id, value: act}
	}
}

// Backward pass through the unit. Waits for gradients from all downstream
// connections, back-propagates, and updates weights.
func (u *Unit) backward(lr float64) {
	grad := 0.0
	needRecv := make(map[string]bool)
	for k := range u.output {
		needRecv[k] = true
	}
	for len(needRecv) > 0 {
		// Get a grad from one of the output connections.
		s := <-u.inputB
		// Accumulate gradient wrt output.
		grad += s.value
		delete(needRecv, s.id)
	}

	// Chain rule through ReLU. If the unit didn't "fire", no real gradients. But
	// still need to do backprop for synchronization purposes.
	if u.rectify && u.preact <= 0 {
		grad = 0.0
	}

	// Update the bias.
	u.bias -= lr * grad

	// Backpropagate and update the weights.
	for k := range u.outputB {
		u.outputB[k] <- signal{id: u.id, value: grad * u.weight[k]}
		// TODO: Might want more flexible control over weight updates.
		u.weight[k] -= lr * grad * u.value[k]
	}
}

// Start starts an endless loop of forward/backward passes.
func (u *Unit) Start(lr float64) {
	for {
		u.forward()
		u.backward(lr)
	}
}
