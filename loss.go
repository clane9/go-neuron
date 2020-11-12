package neuron

import (
	"fmt"
	"math"
)

// MarginLoss computes the maximum-margin SVM loss and its derivative.
func MarginLoss(score float64, target int) (loss float64, grad float64) {
	if !(target == 1 || target == -1) {
		panic(fmt.Sprintf("Expected target +/- 1; got %d", target))
	}

	targetf := float64(target)
	loss = math.Max(1.0-score*targetf, 0.0)
	if score*targetf >= 1.0 {
		grad = 0.0
	} else {
		grad = -targetf
	}
	return
}
