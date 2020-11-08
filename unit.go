package neuron

import (
	"fmt"
	"reflect"
	"time"
)

// A Unit is a single neuron unit with weights input/output channels.
type Unit struct {
	id string
	// Weights for each input connection.
	weight []float64
	bias   float64
	// Values for each input connection.
	value  []float64
	preact float64
	// Accumulated grad wrt output.
	grad float64
	// Two channels for each connection. One forward, one backward.
	input  [][2]chan float64
	output [][2]chan float64
}

// Zero out an earlier activation.
func (u *Unit) zero() {
	u.preact = u.bias
	u.grad = 0.0
	for ii := range u.value {
		u.value[ii] = 0.0
	}
}

// Forward pass through the unit.
// Returns once some input has been received, and a timeout exceeded. Before
// returning, it fires an activation is exceeding a threshold.
func (u *Unit) Forward() (closed bool) {
	// Generate the list of cases to select from. One for each input, plus a
	// default case.
	numIn := len(u.input)
	cases := make([]reflect.SelectCase, numIn+1)
	for ii := range u.input {
		cases[ii] = reflect.SelectCase{
			Dir:  reflect.SelectRecv,
			Chan: reflect.ValueOf(u.input[ii][0]),
		}
	}
	cases[numIn] = reflect.SelectCase{Dir: reflect.SelectDefault}

	numRecv := 0
	waitCount := 0
	numOpen := numIn
	u.zero()

	for {
		chosen, value, ok := reflect.Select(cases)

		if chosen < numIn {
			if ok {
				// Received a signal from one of the input connections.
				// Update the value and pre-activation.
				u.value[chosen] += value.Float()
				u.preact += u.weight[chosen] * value.Float()
				numRecv++
			} else {
				// One of the input connections is now closed. This should only happen
				// when all inputs are closing and the network is shutting down.
				cases[chosen].Chan = reflect.ValueOf(nil)
				numOpen--

				if numOpen <= 0 {
					break
				}
			}
		} else {
			// Default case. Wait for some number of default case hits after the first
			// input before exiting.
			if numRecv > 0 {
				waitCount++
			}
			// TODO: Make these parameters.
			if waitCount >= 5 {
				break
			} else {
				time.Sleep(time.Millisecond)
			}
		}
	}

	// Fire if pre-activation exceeds threshold.
	if u.preact > 0 {
		for ii := range u.output {
			u.output[ii][0] <- u.preact
		}
	}
	closed = numOpen <= 0
	return
}

// Backward pass through the unit.
// Waits for gradients from all downstream connections, back-propagates, and
// updates weights.
func (u *Unit) Backward(lr float64) {
	// If the unit didn't fire, there won't be any gradients.
	// TODO: Need to support more general activations besides ReLU perhaps.
	if u.preact <= 0 {
		return
	}

	// Generate the list of cases to select from. One for each output. No default
	// needed, since I assume Backward only gets called when we expect gradients.
	numOut := len(u.output)
	cases := make([]reflect.SelectCase, numOut)
	for ii := range u.output {
		cases[ii] = reflect.SelectCase{
			Dir:  reflect.SelectRecv,
			Chan: reflect.ValueOf(u.output[ii][1]),
		}
	}

	numRecv := 0
	recvFrom := make(map[int]bool)
	for numRecv < numOut {
		chosen, value, ok := reflect.Select(cases)
		if !ok {
			// Backward connection should never be closed.
			panic(fmt.Sprintf("(%s) Connection %d unexpectedly closed", u.id,
				chosen))
		}

		// Received a grad from one of the output connections.
		// Check that we haven't received from this connection yet.
		if recvFrom[chosen] {
			panic(fmt.Sprintf("(%s) Already received grad from %d", u.id, chosen))
		}
		recvFrom[chosen] = true
		numRecv++

		// Accumulate gradient wrt output.
		u.grad += value.Float()
	}

	// Backpropagate and update the weights.
	for ii := range u.input {
		u.input[ii][1] <- u.grad * u.weight[ii]
		// TODO: Might want more general control over weight updates.
		u.weight[ii] -= lr * u.grad * u.value[ii]
	}
}
