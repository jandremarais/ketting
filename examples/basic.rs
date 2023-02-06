use ketting::{Mlp, Value};

fn main() {
    let m = Mlp::new(3, &[4, 4, 1]);

    let xs = vec![
        vec![Value::new(2.0), Value::new(3.0), Value::new(-1.0)],
        vec![Value::new(3.0), Value::new(-1.0), Value::new(0.5)],
        vec![Value::new(1.5), Value::new(1.0), Value::new(1.0)],
        vec![Value::new(1.0), Value::new(1.0), Value::new(-1.0)],
    ];
    let ys = vec![
        Value::new(1.0),
        Value::new(-1.0),
        Value::new(-1.0),
        Value::new(1.0),
    ];

    for k in 0..40 {
        let ypreds: Vec<Vec<Value>> = xs.iter().map(|x| m.forward(x)).collect();
        let res: Vec<Value> = ys
            .iter()
            .zip(ypreds.iter())
            .map(|(y, pred)| (y.clone() - pred[0].clone()).pow(2))
            .collect();
        let loss = res.iter().fold(Value::new(0.0), |a, b| a + b.clone());
        println!("{} loss: {}", k, loss);
        for p in m.parameters() {
            p.set_grad(0.0);
        }
        loss.backward();
        for p in m.parameters() {
            *p.borrow_data_mut() -= 0.05 * p.borrow_grad();
        }
    }
}
