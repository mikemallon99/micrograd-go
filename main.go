package microgradgo

import (
	"fmt"
	"math"
	"math/rand"
)

type Value struct {
	Data float64
	Grad float64

	backward_fn func()
	prev        []*Value
	op          string
}

func MakeValue(data float64, children []*Value, op string) *Value {
	return &Value{
		Data:        data,
		Grad:        0.0,
		backward_fn: nil,
		prev:        children,
		op:          op,
	}
}

func (v *Value) String() string {
	return fmt.Sprintf("Value: {data=%v, grad=%v, op=%v}", v.Data, v.Grad, v.op)
}

func (v *Value) add(other *Value) *Value {
	out := MakeValue(v.Data+other.Data, []*Value{v, other}, "+")

	out.backward_fn = func() {
		v.Grad += out.Grad
		other.Grad += out.Grad
	}
	return out
}

func (v *Value) mul(other *Value) *Value {
	out := MakeValue(v.Data*other.Data, []*Value{v, other}, "*")

	out.backward_fn = func() {
		v.Grad += other.Grad * out.Grad
		other.Grad += v.Grad * out.Grad
	}
	return out
}

func (v *Value) pow(other float64) *Value {
	out := MakeValue(math.Pow(v.Data, other), []*Value{v}, fmt.Sprintf("**%v", other))

	out.backward_fn = func() {
		v.Grad += (other * math.Pow(v.Data, other-1))
	}
	return out
}

func (v *Value) relu() *Value {
	data := v.Data
	if v.Data < 0.0 {
		data = 0.0
	}
	out := MakeValue(data, []*Value{v}, "ReLU")

	out.backward_fn = func() {
		grad := 1.0
		if out.Data < 0.0 {
			grad = 0.0
		}
		v.Grad += grad * out.Grad
	}
	return out
}

func (v *Value) backward() {
	topo := []*Value{}
	visited := map[*Value]bool{}

	var build_topo func(*Value)

	build_topo = func(v *Value) {
		if !visited[v] {
			visited[v] = true
			for child_idx := range v.prev {
				build_topo(v.prev[child_idx])
			}
			topo = append(topo, v)
		}
	}
	build_topo(v)

	v.Grad = 1.0
	for i := len(topo) - 1; i >= 0; i-- {
		if topo[i].backward_fn != nil {
			topo[i].backward_fn()
		}
	}
}

type Module interface {
	zero_grad()
	parameters() []*Value
}

type Neuron struct {
	w      []*Value
	b      *Value
	nonlin bool
}

func MakeNeuron(nin int, nonlin bool) *Neuron {
	w := []*Value{}
	for _ = range nin {
		w = append(w, MakeValue(rand.Float64()*2-1, []*Value{}, ""))
	}
	return &Neuron{
		w:      w,
		b:      MakeValue(0.0, []*Value{}, ""),
		nonlin: nonlin,
	}
}

func (n *Neuron) forward(x []*Value) *Value {
	act := MakeValue(0.0, []*Value{}, "")
	for i := range x {
		act = act.add(n.w[i].mul(x[i]))
	}
	act = act.add(n.b)
	if n.nonlin {
		return act.relu()
	} else {
		return act
	}
}

func (n *Neuron) parameters() []*Value {
	return append(n.w, n.b)
}

func (n *Neuron) String() string {
	var neuron_type string
	if n.nonlin {
		neuron_type = "ReLU"
	} else {
		neuron_type = "Linear"
	}
	return fmt.Sprintf("%vNeuron(%v)", neuron_type, len(n.w))
}
