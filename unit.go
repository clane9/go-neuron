package neuron

import (
	"math"
	"math/rand"
)

// A Unit is an abstract single neuron unit with Forward, Backward, and Step
// methods.
type Unit interface {
	Forward()
	Backward()
	Step(lr float64)
}

// A HiddenUnit is a single neuron unit belonging in hidden layers with weights
// and input/output channels for forward and backward.
type HiddenUnit struct {
	ID     string
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

// An InputUnit is a single neuron unit belonging in input layers. It has no
// weights, only inputs and outputs.
type InputUnit struct {
	ID      string
	preact  float64
	Input   chan float64
	Output  map[string](chan Signal)
	InputB  chan Signal
	OutputB chan float64
}

// An OutputUnit is a single neuron unit belonging in output layers. It has
// weights, but only a single output channel.
type OutputUnit struct {
	ID         string
	preact     float64
	Weight     map[string]float64
	Bias       float64
	value      map[string]float64
	gradWeight map[string]float64
	gradBias   float64
	// Single Input and Output channels.
	Input   chan Signal
	Output  chan float64
	InputB  chan float64
	OutputB map[string](chan Signal)
}

// A Signal is used to communicate between neuron Units. They contain a value
// and the ID of the sender.
type Signal struct {
	ID    string
	Value float64
}

// NewHiddenUnit creates a new HiddenUnit with a given string id. It allocates
// new input channels and empty maps for weights, values, and outputs.
func NewHiddenUnit(id string) *HiddenUnit {
	u := HiddenUnit{
		ID:         id,
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
	Logf(2, "New hidden unit %s\n", id)
	return &u
}

// NewInputUnit creates a new InputUnit with a given string id.
func NewInputUnit(id string) *InputUnit {
	u := InputUnit{
		ID:      id,
		Input:   make(chan float64, 1),
		Output:  make(map[string](chan Signal)),
		InputB:  make(chan Signal, 512),
		OutputB: make(chan float64, 1),
	}
	Logf(2, "New input unit %s\n", id)
	return &u
}

// NewOutputUnit creates a new OutputUnit with a given string id.
func NewOutputUnit(id string) *OutputUnit {
	u := OutputUnit{
		ID:         id,
		Weight:     make(map[string]float64),
		Bias:       0.0,
		value:      make(map[string]float64),
		gradWeight: make(map[string]float64),
		Input:      make(chan Signal, 512),
		Output:     make(chan float64, 1),
		InputB:     make(chan float64, 1),
		OutputB:    make(map[string](chan Signal)),
	}
	Logf(2, "New output unit %s\n", id)
	return &u
}

// Connect connects two units together in series: u1 -> u2.
// TODO: these turn out to all be the same. Can we consolidate?
func Connect(u1, u2 *HiddenUnit) {
	// Create forward connection from u1 -> u2 by giving u1 a reference to u2's
	// input channel.
	u1.Output[u2.ID] = u2.Input
	// Initialize a weight value for u1 -> u2.
	u2.Weight[u1.ID] = initWeight()
	// Create backward connection from u1 <- u2 by giving u2 a reference to u1's
	// backward input channel.
	u2.OutputB[u1.ID] = u1.InputB
	Logf(2, "Connect: %s -> %s\n", u1.ID, u2.ID)
}

// FeedIn connects an input unit to a hidden unit.
func FeedIn(u1 *InputUnit, u2 *HiddenUnit) {
	u1.Output[u2.ID] = u2.Input
	u2.Weight[u1.ID] = initWeight()
	u2.OutputB[u1.ID] = u1.InputB
	Logf(2, "Feed in: %s -> %s\n", u1.ID, u2.ID)
}

// FeedOut connects a hidden unit to an output unit.
func FeedOut(u1 *HiddenUnit, u2 *OutputUnit) {
	u1.Output[u2.ID] = u2.Input
	u2.Weight[u1.ID] = initWeight()
	u2.OutputB[u1.ID] = u1.InputB
	Logf(2, "Feed out: %s -> %s\n", u1.ID, u2.ID)
}

// Initialize a weight value by sampling randomly from [-0.01, 0.01).
func initWeight() float64 {
	w := rand.Float64()
	w = 0.02*w - 0.01
	return w
}

// Forward pass for hidden units. Collects input from all incoming units and
// fires an activation.
func (u *HiddenUnit) Forward() {
	// Zero out the previous activations, and note the units we need activations
	// from.
	needRecv := make(map[string]bool)
	for k := range u.Weight {
		u.value[k] = 0.0
		needRecv[k] = true
	}
	// Initialize the pre-activation
	u.preact = u.Bias
	// Get inputs from all incoming units.
	for len(needRecv) > 0 {
		s := <-u.Input
		// Update the value and pre-activation.
		u.value[s.ID] += s.Value
		u.preact += u.Weight[s.ID] * s.Value
		delete(needRecv, s.ID)
		Logf(3, "Recv %s -> %s (%.3e)\n", s.ID, u.ID, s.Value)
	}

	// Apply ReLU and fire activation.
	act := math.Max(u.preact, 0.0)
	for k := range u.Output {
		u.Output[k] <- Signal{ID: u.ID, Value: act}
		Logf(3, "Send %s -> %s (%.3e)\n", u.ID, k, act)
	}
}

// Forward pass for input units.
func (u *InputUnit) Forward() {
	// Get single input value and broadcast to all downstream units.
	u.preact = <-u.Input
	Logf(3, "Recv input -> %s (%.3e)\n", u.ID, u.preact)
	for k := range u.Output {
		u.Output[k] <- Signal{ID: u.ID, Value: u.preact}
		Logf(3, "Send %s -> %s (%.3e)\n", u.ID, k, u.preact)
	}
}

// Forward pass for output units.
func (u *OutputUnit) Forward() {
	needRecv := make(map[string]bool)
	for k := range u.Weight {
		u.value[k] = 0.0
		needRecv[k] = true
	}
	u.preact = u.Bias
	// Get inputs from all incoming units and update the values and
	// pre-activation.
	for len(needRecv) > 0 {
		s := <-u.Input
		u.value[s.ID] += s.Value
		u.preact += u.Weight[s.ID] * s.Value
		delete(needRecv, s.ID)
		Logf(3, "Recv %s -> %s (%.3e)\n", s.ID, u.ID, s.Value)
	}

	// Fire activation
	u.Output <- u.preact
	Logf(3, "Send %s -> output (%.3e)\n", u.ID, u.preact)
}

// Backward pass for hidden units. Waits for gradients from all downstream
// connections, updates weight gradients, and back-propagates.
func (u *HiddenUnit) Backward() {
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
		Logf(3, "Recv grad %s -> %s (%.3e)\n", s.ID, u.ID, s.Value)
	}

	// Chain rule through ReLU.
	if u.preact <= 0 {
		grad = 0.0
		Logf(3, "Zero grad; ReLU")
	}

	// If the unit didn't "fire", no real gradients. But still need to do backprop
	// for synchronization purposes.
	for k := range u.Weight {
		u.gradWeight[k] += grad * u.value[k]
		u.OutputB[k] <- Signal{ID: u.ID, Value: grad * u.Weight[k]}
		Logf(3, "Send grad %s -> %s (%.3e)\n", u.ID, k, grad*u.Weight[k])
	}
	u.gradBias += grad
}

// Backward pass for input units. Accumulates gradients wrt inputs and forwards
// to output channel. Used to signal end of backward pass.
func (u *InputUnit) Backward() {
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
		Logf(3, "Recv grad %s -> %s (%.3e)\n", s.ID, u.ID, s.Value)
	}

	// Send out accumulated grad.
	u.OutputB <- grad
	Logf(3, "Send grad %s -> output (%.3e)\n", u.ID, grad)
}

// Backward pass for output units.
func (u *OutputUnit) Backward() {
	// Get a grad from the (only) output connection.
	grad := <-u.InputB
	Logf(3, "Recv grad loss -> %s (%.3e)\n", u.ID, grad)

	for k := range u.Weight {
		u.gradWeight[k] += grad * u.value[k]
		u.OutputB[k] <- Signal{ID: u.ID, Value: grad * u.Weight[k]}
		Logf(3, "Send grad %s -> %s (%.3e)\n", u.ID, k, grad*u.Weight[k])
	}
	u.gradBias += grad
}

// Step for hidden units. Updates weights and bias with negative gradient step.
// TODO: There's currently nothing to make sure that we finish a step before the
// next forward starts.
func (u *HiddenUnit) Step(lr float64) {
	for k := range u.Weight {
		// TODO: Might want to generalize this to other optimizer updates.
		u.Weight[k] -= lr * u.gradWeight[k]
		u.gradWeight[k] = 0.0
	}
	u.Bias -= lr * u.gradBias
	u.gradBias = 0.0
	Logf(3, "Step %s\n", u.ID)
}

// Step for input units. (Do nothing.)
func (u *InputUnit) Step(lr float64) {}

// Step for output units. (Same as for hidden.)
func (u *OutputUnit) Step(lr float64) {
	for k := range u.Weight {
		u.Weight[k] -= lr * u.gradWeight[k]
		u.gradWeight[k] = 0.0
	}
	u.Bias -= lr * u.gradBias
	u.gradBias = 0.0
	Logf(3, "Step %s\n", u.ID)
}
