package neuron

import (
	"math"
	"math/rand"
)

// A Unit is a single neuron unit with weights and input/output channels for
// forward and backward.
type Unit struct {
	ID     string
	Layer  UnitLayer
	preact float64
	// Weights for each input connection.
	Weight map[string]float64
	Bias   float64
	// Values for each input connection.
	value map[string]float64
	// Accumulated gradients for weights and bias.
	gradWeight map[string]float64
	gradBias   float64
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

// A UnitLayer defines the type of layer that a unit belongs to.
type UnitLayer int

const (
	// InputLayer is for input units (no weights)
	InputLayer UnitLayer = iota
	// HiddenLayer is for hidden units (weights and ReLU)
	HiddenLayer
	// OutputLayer is for output units (weights and no ReLU)
	OutputLayer
)

// signals are used to communicate between neuron Units.
type signal struct {
	id    string
	value float64
}

// special ID for input and output channels.
const inputID = "INPUT"
const outputID = "OUTPUT"

// Create a new Unit with a given string id and layer type.
func newUnit(id string, layer UnitLayer, stepDone chan int) *Unit {
	// Set defaults depending on layer.
	bias := 0.0
	inCap := 512
	inCapB := 512
	switch layer {
	case InputLayer:
		inCap = 1
	case HiddenLayer:
		bias = 0.1
	case OutputLayer:
		inCapB = 1
	}

	u := Unit{
		ID:         id,
		Layer:      layer,
		Weight:     make(map[string]float64),
		Bias:       bias,
		value:      make(map[string]float64),
		gradWeight: make(map[string]float64),
		input:      make(chan signal, inCap),
		output:     make(map[string](chan signal)),
		inputB:     make(chan signal, inCapB),
		outputB:    make(map[string](chan signal)),
		stepDone:   stepDone,
	}

	// Add a special output channel.
	if layer == OutputLayer {
		u.output[outputID] = make(chan signal, 1)
	}
	Logf(2, "New unit %s\n", id)
	return &u
}

// connect two units together in series: u1 -> u2.
func connect(u1, u2 *Unit) {
	// Create forward connection from u1 -> u2 by giving u1 a reference to u2's
	// input channel.
	u1.output[u2.ID] = u2.input
	// Initialize a weight value for u1 -> u2.
	u2.Weight[u1.ID] = initWeight()
	// Create backward connection from u1 <- u2 by giving u2 a reference to u1's
	// backward input channel.
	u2.outputB[u1.ID] = u1.inputB
	Logf(2, "Connect: %s -> %s\n", u1.ID, u2.ID)
}

// Initialize a weight value by sampling randomly from [-0.01, 0.01).
func initWeight() float64 {
	w := rand.Float64()
	w = 0.02*w - 0.01
	return w
}

// Forward pass through the unit. Collects input from all incoming units and
// fires an activation.
func (u *Unit) forward() {
	if u.Layer == InputLayer {
		s := <-u.input
		u.preact = s.value
		Logf(3, "Recv input -> %s (%.3e)\n", u.ID, s.value)
	} else {
		// Zero out the previous activations, and note the units we need activations
		// from.
		needRecv := make(map[string]bool)
		for k := range u.Weight {
			u.value[k] = 0.0
			needRecv[k] = true
		}
		// Initialize the pre-activation
		u.preact = u.Bias
		// Get inputs from all incoming units and add up pre-activation.
		for len(needRecv) > 0 {
			s := <-u.input
			u.value[s.id] += s.value
			u.preact += u.Weight[s.id] * s.value
			delete(needRecv, s.id)
			Logf(3, "Recv %s -> %s (%.3e)\n", s.id, u.ID, s.value)
		}
	}

	// Fire activation
	act := u.preact
	if u.Layer == HiddenLayer {
		// Apply ReLU
		act = math.Max(act, 0.0)
	}
	s := signal{id: u.ID, value: act}
	if u.Layer == OutputLayer {
		u.output[outputID] <- s
		Logf(3, "Send %s -> output (%.3e)\n", u.ID, act)
	} else {
		for k := range u.output {
			u.output[k] <- s
			Logf(3, "Send %s -> %s (%.3e)\n", u.ID, k, act)
		}
	}
}

// Backward pass through the unit. Waits for gradients from all downstream
// connections, updates weight gradients, and back-propagates.
func (u *Unit) backward() {
	var grad float64
	if u.Layer == OutputLayer {
		s := <-u.inputB
		grad = s.value
		Logf(3, "Recv grad loss -> %s (%.3e)\n", u.ID, grad)
	} else {
		grad = 0.0
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
			Logf(3, "Recv grad %s -> %s (%.3e)\n", s.id, u.ID, s.value)
		}
	}

	// Backprop. If the unit didn't "fire", no real gradients. But still need to
	// do backprop for synchronization purposes.
	if u.Layer != InputLayer {
		// Chain rule for ReLU.
		if u.Layer == HiddenLayer && u.preact <= 0 {
			grad = 0.0
		}
		// Chain rule for weights.
		for k := range u.Weight {
			u.gradWeight[k] += grad * u.value[k]
			u.outputB[k] <- signal{id: u.ID, value: grad * u.Weight[k]}
			Logf(3, "Send grad %s -> %s (%.3e)\n", u.ID, k, grad*u.Weight[k])
		}
		u.gradBias += grad
	}
}

// Update the weights and bias by taking a gradient descent step.
func (u *Unit) step(lr float64) {
	if u.Layer != InputLayer {
		for k := range u.Weight {
			// TODO: Might want to generalize this to other optimizer updates.
			u.Weight[k] -= lr * u.gradWeight[k]
			u.gradWeight[k] = 0.0
		}
		u.Bias -= lr * u.gradBias
		u.gradBias = 0.0
	}
	Logf(3, "Step %s\n", u.ID)
}

// Start starts an endless loop of forward and backward passes with periodic
// gradient updates.
func (u *Unit) start(train bool, updateFreq int, lr float64) {
	step := 1
	for {
		u.forward()
		if train {
			u.backward()
			if updateFreq > 0 && step%updateFreq == 0 {
				u.step(lr)
			}
			u.stepDone <- 1
		}
		step++
	}
}
