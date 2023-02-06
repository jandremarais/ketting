use std::{
    cell::{RefCell, RefMut},
    collections::{HashSet, VecDeque},
    fmt,
    iter::Sum,
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
};

use rand::Rng;
use uuid::Uuid;

#[cfg(test)]
mod tests;
#[derive(Clone, Debug, PartialEq)]
enum Operation {
    Add,
    Mul,
    Tanh,
    Exp,
    Pow,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Value {
    id: Uuid,
    pub data: Rc<RefCell<f32>>,
    pub grad: Rc<RefCell<f32>>,
    pub children: Vec<Self>,
    op: Option<Operation>,
    label: Option<String>,
}

impl Value {
    pub fn new(data: f32) -> Self {
        Self {
            id: Uuid::new_v4(),
            data: Rc::new(RefCell::new(data)),
            grad: Rc::new(RefCell::new(0.0)),
            children: Vec::new(),
            op: None,
            label: None,
        }
    }
    pub fn borrow_data(&self) -> f32 {
        *self.data.borrow()
    }
    pub fn borrow_data_mut(&self) -> RefMut<f32> {
        self.data.borrow_mut()
    }
    pub fn borrow_grad(&self) -> f32 {
        *self.grad.borrow()
    }
    pub fn borrow_grad_mut(&self) -> f32 {
        *self.grad.borrow_mut()
    }
    pub fn set_grad(&self, grad: f32) {
        *self.grad.borrow_mut() = grad;
    }
    pub fn update_grad(&self, grad: f32) {
        *self.grad.borrow_mut() += grad;
    }

    pub fn tanh(&self) -> Self {
        let n = self.borrow_data();
        let tanh = ((2.0 * n).exp() - 1.0) / ((2.0 * n).exp() + 1.0);
        let mut v = Value::new(tanh);
        v.children.push(self.clone());
        v.op = Some(Operation::Tanh);
        v
    }

    pub fn exp(&self) -> Self {
        let n = self.borrow_data();
        let mut v = Value::new(n.exp());
        v.children.push(self.clone());
        v.op = Some(Operation::Exp);
        v
    }

    pub fn pow(&self, rhs: i32) -> Self {
        let n = self.borrow_data();
        // WARNING: Not sure treating it as Value is right
        // Karpathy left it as a 'scalar'
        let rhs_val = Value::new(rhs as f32);
        let mut v = Value::new(n.powi(rhs));
        v.children.push(self.clone());
        v.children.push(rhs_val);
        v.op = Some(Operation::Pow);
        v
    }

    fn backward_local(&self) {
        if let Some(operation) = &self.op {
            match operation {
                Operation::Add => {
                    let new_grad = 1.0 * self.borrow_grad();
                    self.children[0].update_grad(new_grad);
                    self.children[1].update_grad(new_grad);
                }
                Operation::Mul => {
                    let new_grad = self.children[1].borrow_data() * self.borrow_grad();
                    self.children[0].update_grad(new_grad);
                    let new_grad = self.children[0].borrow_data() * self.borrow_grad();
                    self.children[1].update_grad(new_grad);
                }
                Operation::Tanh => {
                    let new_grad = (1.0 - self.borrow_data().powi(2)) * self.borrow_grad();
                    self.children[0].update_grad(new_grad);
                }
                Operation::Exp => {
                    let new_grad = self.borrow_data() * self.borrow_grad();
                    self.children[0].update_grad(new_grad);
                }
                Operation::Pow => {
                    // x^n -> n*x^(n-1)
                    let new_grad = self.children[1].borrow_data()
                        * self.children[0]
                            .borrow_data()
                            .powf(self.children[1].borrow_data() - 1.0)
                        * self.borrow_grad();
                    self.children[0].update_grad(new_grad);
                    // WARNING: nothing happening to children[1]
                }
            }
        }
    }

    pub fn backward(&self) {
        self.set_grad(1.0);

        let topo = build_topo(&self);
        for v in topo.iter().rev() {
            v.backward_local();
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Value(data={}, label={}, grad={})",
            self.borrow_data(),
            self.label.clone().unwrap_or(String::new()),
            self.borrow_grad()
        )
    }
}

impl Add for Value {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let mut v = Self::new(*self.data.borrow() + *rhs.data.borrow());
        v.children.push(self.clone());
        v.children.push(rhs.clone());
        v.op = Some(Operation::Add);
        v
    }
}

impl Add<f32> for Value {
    type Output = Self;
    fn add(self, rhs: f32) -> Self::Output {
        self + Value::new(rhs)
    }
}

impl Add<Value> for f32 {
    type Output = Value;
    fn add(self, rhs: Value) -> Self::Output {
        rhs + self
    }
}

impl Mul for Value {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut v = Self::new(*self.data.borrow() * *rhs.data.borrow());
        v.children.push(self.clone());
        v.children.push(rhs.clone());
        v.op = Some(Operation::Mul);
        v
    }
}

impl Mul<f32> for Value {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self::Output {
        self * Value::new(rhs)
    }
}

impl Div for Value {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.pow(-1)
    }
}

impl Sub for Value {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl Sub<f32> for Value {
    type Output = Self;
    fn sub(self, rhs: f32) -> Self::Output {
        self + (-rhs)
    }
}

impl Neg for Value {
    type Output = Self;
    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

// impl Sum for Value {
//     fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
//         let out = iter.next().unwrap();
//         // let out = Value::new(0.0);
//         iter.fold(out, |a, b| a + b)
//     }
// }

fn build_topo(v: &Value) -> Vec<&Value> {
    let mut collected: HashSet<Uuid> = HashSet::new();
    let mut queue: VecDeque<&Value> = VecDeque::new();
    queue.push_front(v);
    let mut topo = Vec::new();

    while let Some(value) = queue.pop_front() {
        let mut can_collect = true;
        for child in value.children.iter() {
            if !collected.contains(&child.id) {
                queue.push_front(value);
                queue.push_front(child);
                can_collect = false;
                break;
            }
        }

        if can_collect {
            collected.insert(value.id);
            topo.push(value);
        }
    }
    topo
}

pub struct Neuron {
    pub nin: usize,
    pub w: Vec<Value>,
    pub b: Value,
}

impl Neuron {
    fn new(nin: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            nin,
            w: (0..nin)
                .map(|_| Value::new(rng.gen_range(-1.0..1.0)))
                .collect(),
            b: Value::new(rng.gen_range(-1.0..1.0)),
        }
    }

    fn forward(&self, x: &[Value]) -> Value {
        let xw: Vec<Value> = x
            .iter()
            .zip(self.w.iter())
            .map(|(xi, wi)| xi.clone() * wi.clone())
            .collect();

        let out = xw.into_iter().fold(self.b.clone(), |a, b| a + b);
        out.tanh()
    }

    fn parameters(&self) -> Vec<&Value> {
        let b = vec![&self.b];
        self.w.iter().chain(b).collect()
    }
}

pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    fn new(nin: usize, nout: usize) -> Self {
        Self {
            neurons: (0..nout).map(|_| Neuron::new(nin)).collect(),
        }
    }

    fn forward(&self, x: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forward(x)).collect()
    }

    fn parameters(&self) -> Vec<&Value> {
        let mut params = Vec::new();
        for n in self.neurons.iter() {
            params.extend(n.parameters())
        }
        params
    }
}

pub struct Mlp {
    pub layers: Vec<Layer>,
}

impl Mlp {
    pub fn new(nin: usize, nouts: &[usize]) -> Self {
        let mut sizes = vec![nin];
        sizes.extend(nouts);
        let layers = sizes.windows(2).map(|w| Layer::new(w[0], w[1])).collect();
        Self { layers }
    }

    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        let mut out: Vec<Value> = x.into();
        for l in self.layers.iter() {
            out = l.forward(&out);
        }
        out
    }

    pub fn parameters(&self) -> Vec<&Value> {
        let mut params = Vec::new();
        for n in self.layers.iter() {
            params.extend(n.parameters())
        }
        params
    }
}
