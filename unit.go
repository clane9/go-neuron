package neuron

import (
	"math"
	"math/rand"
)

// A Unit is a single neuron unit with weights input/output channels.
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
	// Single Input channel.
	Input chan Signal
	// Output channels for each downstream connection.
	Output map[string](chan Signal)
	// Similarly, input and output channels for backwards communication.
	InputB  chan Signal
	OutputB map[string](chan Signal)
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

// A Signal is used to communicate between neuron Units. They contain a value
// and the ID of the sender.
type Signal struct {
	ID    string
	Value float64
}

// OutputChanID is the label used to identify output layer units' output channel.
const OutputChanID = "OUTPUT"

// NewUnit creates a new Unit with a given string id. It allocates new input
// channels and empty maps for weights, values, and outputs.
func NewUnit(id string, layer UnitLayer) *Unit {
	u := Unit{
		ID:         id,
		Layer:      layer,
		Weight:     make(map[string]float64),
		Bias:       0.1,
		value:      make(map[string]float64),
		gradWeight: make(map[string]float64),
		// TODO: Need a large buffer to accommodate multiple units sending signals
		// simultaneously. But how big do I need?
		Input:   make(chan Signal, 512),
		Output:  make(map[string](chan Signal)),
		InputB:  make(chan Signal, 512),
		OutputB: make(map[string](chan Signal)),
	}
	// Add a special output channel for output units.
	if layer == OutputLayer {
		u.Output[OutputChanID] = make(chan Signal, 1)
	}
	return &u
}

// Connect connects two units together in series: u1 -> u2.
func Connect(u1, u2 *Unit) {
	// Create forward connection from u1 -> u2 by giving u1 a reference to u2's
	// input channel.
	u1.Output[u2.ID] = u2.Input
	// Initialize a weight value for u1 -> u2.
	u2.initWeight(u1.ID)
	// Create backward connection from u1 <- u2 by giving u2 a reference to u1's
	// backward input channel.
	u2.OutputB[u1.ID] = u1.InputB
}

var rng = rand.New(rand.NewSource(12))

// Initialize a weight value by sampling randomly from [-0.01, 0.01).
func (u *Unit) initWeight(id string) float64 {
	w := rng.Float64()
	w = 0.02*w - 0.01
	u.Weight[id] = w
	return w
}

// Zero out previous activation.
func (u *Unit) zero() {
	for k := range u.value {
		u.value[k] = 0.0
	}
	u.preact = u.Bias
}

// Zero out accumulated gradient.
func (u *Unit) zeroGrad() {
	for k := range u.gradWeight {
		u.gradWeight[k] = 0.0
	}
	u.gradBias = 0.0
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
	if u.Layer == InputLayer {
		s := <-u.Input
		u.preact = s.Value
	} else {
		u.zero()
		needRecv := make(map[string]bool)
		for k := range u.Weight {
			needRecv[k] = true
		}
		for len(needRecv) > 0 {
			// Wait for a signal from one of the input connections.
			// Update the value and pre-activation.
			s := <-u.Input
			u.value[s.ID] += s.Value
			u.preact += u.Weight[s.ID] * s.Value
			delete(needRecv, s.ID)
		}
	}

	// Fire activation
	act := u.preact
	if u.Layer == HiddenLayer {
		act = math.Max(act, 0.0)
	}
	for k := range u.Output {
		u.Output[k] <- Signal{ID: u.ID, Value: act}
	}
}

// Backward pass through the unit. Waits for gradients from all downstream
// connections, updates weight gradients, and back-propagates.
func (u *Unit) backward() {
	grad := 0.0
	needRecv := make(map[string]bool)
	for k := range u.Output {
		needRecv[k] = true
	}
	for len(needRecv) > 0 {
		// Get a grad from one of the output connections.
		s := <-u.InputB
		// Accumulate gradient wrt output.
		grad += s.Value
		delete(needRecv, s.ID)
	}

	// Backprop for units other than inputs. If the unit didn't "fire", no real
	// gradients. But still need to do backprop for synchronization purposes.
	if u.Layer != InputLayer {
		// Chain rule through ReLU.
		if u.Layer == HiddenLayer && u.preact <= 0 {
			grad = 0.0
		}

		for k := range u.Weight {
			u.gradWeight[k] += grad * u.value[k]
			u.OutputB[k] <- Signal{ID: u.ID, Value: grad * u.Weight[k]}
		}
		u.gradBias += grad
	}
}

// Update the weights and bias by taking a gradient descent step.
func (u *Unit) step(lr float64) {
	if u.Layer != InputLayer {
		for k := range u.Weight {
			u.Weight[k] -= lr * u.gradWeight[k]
		}
		u.Bias -= lr * u.gradBias
		u.zeroGrad()
	}
}

// Start starts an endless loop of forward and backward passes with periodic
// gradient updates.
func (u *Unit) Start(train bool, updateFreq int, lr float64) {
	step := 1
	for {
		u.forward()
		if train {
			u.backward()
			if updateFreq > 0 && step%updateFreq == 0 {
				u.step(lr)
			}
		}
		step++
	}
}
