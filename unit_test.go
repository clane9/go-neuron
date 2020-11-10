package neuron_test

import (
	"testing"

	"github.com/clane9/go-neuron"
)

func TestNewUnit(t *testing.T) {
	id := "0000"
	u := neuron.NewUnit(id, neuron.HiddenLayer)
	if u.ID != id {
		t.Errorf("u.ID = %s; expected %s", u.ID, id)
	}
	initBias := 0.1
	if u.Bias != initBias {
		t.Errorf("u.Bias = %.2f; expected %.2f", u.Bias, initBias)
	}
}

func TestConnect(t *testing.T) {
	u1 := neuron.NewUnit("0001", neuron.InputLayer)
	u2 := neuron.NewUnit("0002", neuron.HiddenLayer)
	neuron.Connect(u1, u2)
	if u2.Weight[u1.ID] == 0.0 {
		t.Errorf("u1 -> u2 weight not initialized")
	}
	s1 := neuron.Signal{ID: u1.ID, Value: 0.91}
	u1.Output[u2.ID] <- s1
	s2 := <-u2.Input
	if s1.Value != s2.Value {
		t.Errorf("Output of u1 (%.2f) not equal to input of u2 (%.2f)",
			s1.Value, s2.Value)
	}
}
