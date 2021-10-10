package neuron

// An Optimizer performs gradient based parameter updates
type Optimizer interface {
	Step(id string, p *Param)
	New() Optimizer
}

// SGD Optimizer with momentum and weight decay
type SGD struct {
	Lr          float64
	Momentum    float64
	WeightDecay float64
	buf         map[string]float64
}

// Step takes an SGD optimization step on one scalar parameter. id is used track
// the optimizer state, i.e. momentum buffer, for this parameter.
func (opt *SGD) Step(id string, p *Param) {
	if !p.RequiresGrad {
		return
	}

	grad := p.grad
	if opt.WeightDecay > 0 {
		grad += opt.WeightDecay * p.Data
	}

	var v float64
	var ok bool

	if opt.Momentum > 0 {
		v, ok = opt.buf[id]
		if !ok {
			v = grad
		} else {
			v = opt.Momentum*v + grad
		}
		opt.buf[id] = v
	} else {
		v = grad
	}
	p.Data -= opt.Lr * v
	p.grad = 0.0
}

// New initializes a new SGD optimizer with the same parameters.
func (opt *SGD) New() Optimizer {
	return NewSGD(opt.Lr, opt.Momentum, opt.WeightDecay)
}

// NewSGD creates a new SGD optimizer.
func NewSGD(lr float64, momentum float64, weightDecay float64) *SGD {
	return &SGD{
		Lr:          lr,
		Momentum:    momentum,
		WeightDecay: weightDecay,
		buf:         make(map[string]float64),
	}
}
