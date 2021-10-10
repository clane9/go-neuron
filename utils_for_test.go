package neuron

import (
	"math"
	"testing"
)

// Check that a function panics.
// https://stackoverflow.com/a/31596110
func assertPanic(t *testing.T, f func()) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("The code did not panic")
		}
	}()
	f()
}

// Test whether two values are equal up to a tolerance.
func almostEqual(a, b float64) bool {
	const tol = 1.0e-06
	return math.Abs(a-b)/math.Abs(b) < tol
}
