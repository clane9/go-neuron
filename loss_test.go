package neuron_test

import (
	"testing"

	"github.com/clane9/go-neuron"
)

// Test margin loss.
func TestMarginLoss(t *testing.T) {
	scores := []float64{9.0, 9.0}
	targets := []int{1, -1}
	lossWant := []float64{0.0, 10.0}
	gradWant := []float64{0.0, 1.0}

	for ii := range scores {
		loss, grad := neuron.MarginLoss(scores[ii], targets[ii])
		if loss != lossWant[ii] || grad != gradWant[ii] {
			t.Errorf("(%d) Margin loss returned (%.3f, %.3f); expected (%.3f, %.3f)",
				ii, loss, grad, lossWant[ii], gradWant[ii])
		}
	}

	assertPanic(t, func() { neuron.MarginLoss(1.0, 99) })
}
