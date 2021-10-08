package neuron

// A Param is a neural network parameter
type Param struct {
	Data         float64
	RequiresGrad bool
	value        float64
	grad         float64
}
