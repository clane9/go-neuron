package neuron_test

import (
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/clane9/go-neuron"
)

// Test construction of new input, hidden, and output units.
func TestNewUnits(t *testing.T) {
	const id = "0000"
	const initHBias = 0.1
	const initOBias = 0.0

	// New hidden unit.
	uH := neuron.NewHiddenUnit(id)
	if uH.ID != id {
		t.Errorf("ID = %s; expected %s", uH.ID, id)
	}
	if uH.Bias != initHBias {
		t.Errorf("Bias = %.2f; expected %.2f", uH.Bias, initHBias)
	}

	// New input unit.
	uI := neuron.NewInputUnit(id)
	if uI.ID != id {
		t.Errorf("ID = %s; expected %s", uI.ID, id)
	}

	// New output unit.
	uO := neuron.NewOutputUnit(id)
	if uO.ID != id {
		t.Errorf("ID = %s; expected %s", uO.ID, id)
	}
	if uO.Bias != initOBias {
		t.Errorf("Bias = %.2f; expected %.2f", uO.Bias, initOBias)
	}
}

// Test that we can connect units together and pass signals from one to the
// next. (Bypasses Forward methods.)
func TestConnections(t *testing.T) {
	// Construct a simple series of units.
	u1 := neuron.NewInputUnit("0001")
	u2 := neuron.NewHiddenUnit("0002")
	u3 := neuron.NewHiddenUnit("0003")
	u4 := neuron.NewOutputUnit("0004")

	neuron.FeedIn(u1, u2)
	neuron.Connect(u2, u3)
	neuron.FeedOut(u3, u4)

	// Check that weights are initialized.
	if u2.Weight[u1.ID] == 0.0 ||
		u3.Weight[u2.ID] == 0.0 ||
		u4.Weight[u3.ID] == 0.0 {
		t.Errorf("Not all weights are initialized")
	}

	// Pass a signal through all the channels in the series.
	// (Bypassing Forward methods.)
	const val = 0.91
	s1 := neuron.Signal{ID: u1.ID, Value: val}
	u1.Output[u2.ID] <- s1
	s2 := getSignalTimeout(u2.Input, t)
	u2.Output[u3.ID] <- s2
	s3 := getSignalTimeout(u3.Input, t)
	u3.Output[u4.ID] <- s3
	s4 := getSignalTimeout(u4.Input, t)

	// Check that the values are the same. How could they not be?
	if s2.Value != val ||
		s3.Value != val ||
		s4.Value != val {
		t.Errorf("Not all signal values are equal")
	}
}

// Test Forward/Backward/Step methods sequentially.
func TestForwardBackwardStep(t *testing.T) {
	// Seed rand so we get the same weights.
	rand.Seed(12)

	// Construct a simple series of units.
	u1 := neuron.NewInputUnit("0001")
	u2 := neuron.NewHiddenUnit("0002")
	u3 := neuron.NewHiddenUnit("0003")
	u4 := neuron.NewOutputUnit("0004")
	neuron.FeedIn(u1, u2)
	neuron.Connect(u2, u3)
	neuron.FeedOut(u3, u4)

	// Sequential forward pass.
	const inVal = 0.12
	const outVal = -0.0005913915
	u1.Input <- inVal
	u1.Forward()
	u2.Forward()
	u3.Forward()
	u4.Forward()
	v := <-u4.Output

	if !almostEqual(outVal, v) {
		t.Errorf("Forward pass returned %.10f; expected %.4f", v, outVal)
	}

	// Sequential backward pass.
	const grad = 1.0
	u4.InputB <- grad
	u4.Backward()
	u3.Backward()
	u2.Backward()
	u1.Backward()

	// Sequential gradient steps.
	const lr = 1.0
	u1.Step(lr)
	u2.Step(lr)
	u3.Step(lr)
	u4.Step(lr)

	// Check that weight updates match some expected values.
	checkWeight("u2", u2.Weight[u1.ID], 0.0028339391, t)
	checkWeight("u3", u3.Weight[u2.ID], -0.0043019064, t)
	checkWeight("u4", u4.Weight[u3.ID], -0.1054516327, t)
}

// Wait for a Signal from a channel with a 10 ms timeout.
func getSignalTimeout(c chan neuron.Signal, t *testing.T) neuron.Signal {
	var s neuron.Signal
	select {
	case s = <-c:
	case <-time.After(10 * time.Millisecond):
		t.Fatalf("Signal not received before timeout")
	}
	return s
}

// Check that the updated weight is as expected.
func checkWeight(id string, wGot, wWant float64, t *testing.T) {
	if !almostEqual(wGot, wWant) {
		t.Errorf("Updated %s weight is %.10f; expected %.4f", id, wGot, wWant)
	}
}

// Test whether two values are equal up to a tolerance.
func almostEqual(a, b float64) bool {
	const tol = 1.0e-06
	return math.Abs(a-b) < tol
}
