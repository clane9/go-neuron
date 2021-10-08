package neuron

// A Param is a neural network parameter
type Param struct {
	Data         float64
	RequiresGrad bool
	value        float64
	grad         float64
}

// ZeroGrad zeros out the parameter's gradient
func (p *Param) ZeroGrad() {
	p.grad = 0.0
}
