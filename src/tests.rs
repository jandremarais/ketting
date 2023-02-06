use super::*;

#[test]
fn test_add() {
    let v1 = Value::new(1.0);
    let v2 = Value::new(2.0);
    let v3 = v1.clone() + v2.clone();
    let res = *v3.data.borrow();
    assert_eq!(res, 3.0);
    assert_eq!(v3.children[0], v1);
    assert_eq!(v3.children[1], v2);
    assert_eq!(v3.op, Some(Operation::Add));
}

#[test]
fn test_add_twice() {
    let v1 = Value::new(1.0);
    let v2 = Value::new(2.0);
    let v3 = Value::new(3.0);
    let v1v2 = v1.clone() + v2.clone();
    let v4 = v1v2.clone() + v3.clone();
    let res = *v4.data.borrow();
    assert_eq!(res, 6.0);
    assert_eq!(v4.children.len(), 2);
    assert_eq!(v4.children[0], v1v2);
    assert_eq!(v4.children[1], v3);
}

#[test]
fn test_add_then_mut() {
    let v1 = Value::new(1.0);
    let v2 = Value::new(2.0);
    let v3 = v1.clone() + v2.clone();
    assert_eq!(v3.children[0], v1);
    *v1.data.borrow_mut() += 1.0;
    assert_eq!(v3.children[0], v1);
    let res = *v3.children[0].data.borrow();
    assert_eq!(res, 2.0);
}

#[test]
fn test_mul() {
    let v1 = Value::new(3.0);
    let v2 = Value::new(2.0);
    let v3 = v1.clone() * v2.clone();
    let res = *v3.data.borrow();
    assert_eq!(res, 6.0);
    assert_eq!(v3.children[0], v1);
    assert_eq!(v3.children[1], v2);
    assert_eq!(v3.op, Some(Operation::Mul));
}

#[test]
fn test_tanh() {
    let v1 = Value::new(2.0);
    let v2 = v1.tanh();

    assert!((v2.borrow_data() - 0.96402758).abs() < 0.00001);
    assert_eq!(v2.children[0], v1);
    assert_eq!(v2.children.len(), 1);
    assert_eq!(v2.op, Some(Operation::Tanh));
}

#[test]
fn test_karpathy_expression_output() {
    let a = Value::new(2.0);
    let b = Value::new(-3.0);
    let c = Value::new(10.0);
    let e = a.clone() * b.clone();
    let d = e.clone() + c.clone();
    let f = Value::new(-2.0);
    let loss = d.clone() * f.clone();
    assert_eq!(loss.borrow_data(), -8.0);
}

#[test]
fn test_karpathy_expression_data_nudge() {
    let a = Value::new(2.0);
    let b = Value::new(-3.0);
    let c = Value::new(10.0);
    let e = a.clone() * b.clone();
    let d = e.clone() + c.clone();
    let f = Value::new(-2.0);
    let loss1 = d.clone() * f.clone();

    loss1.set_grad(1.0);
    f.set_grad(4.0);
    d.set_grad(-2.0);
    e.set_grad(-2.0);
    c.set_grad(-2.0);
    a.set_grad(-3.0 * -2.0);
    b.set_grad(2.0 * -2.0);

    let h = 0.01;
    // *a.data.borrow_mut() += 0.1;
    *a.borrow_data_mut() += h * a.borrow_grad();
    *b.borrow_data_mut() += h * b.borrow_grad();
    *c.borrow_data_mut() += h * c.borrow_grad();
    *f.borrow_data_mut() += h * f.borrow_grad();

    let e = a.clone() * b.clone();
    let d = e.clone() + c.clone();
    let loss2 = d.clone() * f.clone();

    assert!(loss2.borrow_data() > loss1.borrow_data());
}

#[test]
fn test_karpathy_neuron_forward() {
    let x1 = Value::new(2.0);
    let x2 = Value::new(0.0);
    let w1 = Value::new(-3.0);
    let w2 = Value::new(1.0);
    let b = Value::new(6.8813735870195432);
    let x1w1 = x1.clone() * w1.clone();
    let x2w2 = x2.clone() * w2.clone();
    let x1w1x2w2 = x1w1.clone() + x2w2.clone();
    let n = x1w1x2w2.clone() + b;
    let o = n.tanh();
    assert!((o.borrow_data() - 0.7071).abs() < 0.00001);
}

#[test]
fn test_karpathy_neuron_grad() {
    let x1 = Value::new(2.0);
    let x2 = Value::new(0.0);
    let w1 = Value::new(-3.0);
    let w2 = Value::new(1.0);
    let b = Value::new(6.8813735870195432);
    let x1w1 = x1.clone() * w1.clone();
    let x2w2 = x2.clone() * w2.clone();
    let x1w1x2w2 = x1w1.clone() + x2w2.clone();
    let n = x1w1x2w2.clone() + b.clone();
    let o = n.tanh();

    o.set_grad(1.0);
    n.set_grad(0.5);
    x1w1x2w2.set_grad(0.5);
    b.set_grad(0.5);

    x2w2.set_grad(0.5);
    x1w1.set_grad(0.5);

    x1.set_grad(-1.5);
    w1.set_grad(1.0);
    x2.set_grad(0.5);
    w2.set_grad(0.0);

    // weight update
    let h = 0.0001;
    *w1.borrow_data_mut() -= w1.borrow_grad() * h;
    *w2.borrow_data_mut() -= w2.borrow_grad() * h;

    let x1w1 = x1.clone() * w1.clone();
    let x2w2 = x2.clone() * w2.clone();
    let x1w1x2w2 = x1w1.clone() + x2w2.clone();
    let n = x1w1x2w2.clone() + b.clone();
    let o2 = n.tanh();

    assert!(o2.borrow_data() < o.borrow_data());
}

#[test]
fn test_karpathy_neuron_backward() {
    let x1 = Value::new(2.0);
    let x2 = Value::new(0.0);
    let w1 = Value::new(-3.0);
    let w2 = Value::new(1.0);
    let b = Value::new(6.8813735870195432);
    let x1w1 = x1.clone() * w1.clone();
    let x2w2 = x2.clone() * w2.clone();
    let x1w1x2w2 = x1w1.clone() + x2w2.clone();
    let n = x1w1x2w2.clone() + b.clone();
    let o = n.tanh();

    o.set_grad(1.0);
    o.backward_local();
    n.backward_local();
    x1w1x2w2.backward_local();
    b.backward_local();

    x2w2.backward_local();
    x1w1.backward_local();

    assert_eq!(o.borrow_grad(), 1.0);
    assert!((n.borrow_grad() - 0.5).abs() < 0.0001);
    assert!((x1w1x2w2.borrow_grad() - 0.5).abs() < 0.0001);
    assert!((b.borrow_grad() - 0.5).abs() < 0.0001);
    assert!((x2w2.borrow_grad() - 0.5).abs() < 0.0001);
    assert!((x1w1.borrow_grad() - 0.5).abs() < 0.0001);
    assert!((x1.borrow_grad() - -1.5).abs() < 0.0001);
    assert!((w1.borrow_grad() - 1.0).abs() < 0.0001);
    assert!((x2.borrow_grad() - 0.5).abs() < 0.0001);
    assert!((w2.borrow_grad() - 0.0).abs() < 0.0001);
}

#[test]
fn test_grad_update_of_child() {
    let x1 = Value::new(2.0);
    let x2 = x1.tanh();

    assert_eq!(x2.children[0], x1);

    x2.children[0].set_grad(2.0);
    assert_eq!(x1.borrow_grad(), 2.0);
}

#[test]
fn test_build_topo() {
    let x1 = Value::new(2.0);
    let x2 = Value::new(1.0);
    let y = x1.clone() + x2.clone();
    let y2 = y.clone() * x1.clone();
    let topo = build_topo(&y2);
    assert_eq!(&y2, topo[3]);
    assert_eq!(&y, topo[2]);
    assert_eq!(&x1, topo[0]);
    assert_eq!(&x2, topo[1]);
}

#[test]
fn test_karpathy_build_topo() {
    let x1 = Value::new(2.0);
    let x2 = Value::new(0.0);
    let w1 = Value::new(-3.0);
    let w2 = Value::new(1.0);
    let b = Value::new(6.8813735870195432);
    let x1w1 = x1.clone() * w1.clone();
    let x2w2 = x2.clone() * w2.clone();
    let x1w1x2w2 = x1w1.clone() + x2w2.clone();
    let n = x1w1x2w2.clone() + b.clone();
    let o = n.tanh();
    let topo = build_topo(&o);

    assert_eq!(topo[9], &o);
    assert_eq!(topo[0], &x1);
}

#[test]
fn test_karpathy_backward() {
    let x1 = Value::new(2.0);
    let x2 = Value::new(0.0);
    let w1 = Value::new(-3.0);
    let w2 = Value::new(1.0);
    let b = Value::new(6.8813735870195432);
    let x1w1 = x1.clone() * w1.clone();
    let x2w2 = x2.clone() * w2.clone();
    let x1w1x2w2 = x1w1.clone() + x2w2.clone();
    let n = x1w1x2w2.clone() + b.clone();
    let o = n.tanh();
    o.backward();

    assert_eq!(o.borrow_grad(), 1.0);
    assert!((n.borrow_grad() - 0.5).abs() < 0.0001);
    assert!((x1w1x2w2.borrow_grad() - 0.5).abs() < 0.0001);
    assert!((b.borrow_grad() - 0.5).abs() < 0.0001);
    assert!((x2w2.borrow_grad() - 0.5).abs() < 0.0001);
    assert!((x1w1.borrow_grad() - 0.5).abs() < 0.0001);
    assert!((x1.borrow_grad() - -1.5).abs() < 0.0001);
    assert!((w1.borrow_grad() - 1.0).abs() < 0.0001);
    assert!((x2.borrow_grad() - 0.5).abs() < 0.0001);
    assert!((w2.borrow_grad() - 0.0).abs() < 0.0001);
}

#[test]
fn test_karpathy_aa_bug() {
    let a = Value::new(3.0);
    let b = a.clone() + a.clone();
    b.backward();
    assert_eq!(a.borrow_grad(), 2.0);
}

#[test]
fn test_karpathy_aa_bug2() {
    let a = Value::new(-2.0);
    let b = Value::new(3.0);
    let d = a.clone() * b.clone(); // -6.0
    let e = a.clone() + b.clone(); // 1.0
    let f = d.clone() * e.clone(); // -6.0
    f.backward();
    assert_eq!(e.borrow_grad(), -6.0);
    assert_eq!(d.borrow_grad(), 1.0);
    assert_eq!(a.borrow_grad(), -3.0); // 3.0 * 1.0 + 1.0 * -6.0
    assert_eq!(b.borrow_grad(), -8.0); // -2.0 * 1.0 + 1.0 * -6.0
}

#[test]
fn test_add_float() {
    let a = Value::new(2.0);
    let b = a.clone() + 3.0;
    b.backward();

    assert_eq!(b.borrow_data(), 5.0);
    assert_eq!(a.borrow_grad(), 1.0);
}

#[test]
fn test_mul_float() {
    let a = Value::new(2.0);
    let b = a.clone() * 3.0;
    b.backward();

    assert_eq!(b.borrow_data(), 6.0);
    assert_eq!(a.borrow_grad(), 3.0);
}

#[test]
fn test_exp() {
    let a = Value::new(2.0);
    let b = a.exp();
    b.backward();

    assert!((b.borrow_data() - 7.389056).abs() < 0.001);
    assert!((a.borrow_grad() - 7.389056).abs() < 0.001);
}

#[test]
fn test_div() {
    let a = Value::new(2.0);
    let b = Value::new(4.0);
    let c = a.clone() / b.clone();
    c.backward();
    assert_eq!(0.5, c.borrow_data());
    assert_eq!(a.borrow_grad(), 1.0 / 4.0);
    assert_eq!(b.borrow_grad(), -2.0 / 4.0_f32.powi(2));
}

#[test]
fn test_sub() {
    let a = Value::new(2.0);
    let b = Value::new(4.0);
    let c = a.clone() - b.clone();
    c.backward();
    assert_eq!(-2.0, c.borrow_data());
    assert_eq!(a.borrow_grad(), 1.0);
    assert_eq!(b.borrow_grad(), -1.0);
}

#[test]
fn test_new_tanh_grad() {
    let n = Value::new(0.8814);
    let tmp = n.clone() * 2.0;
    let e = tmp.exp();
    let z = e.clone() - 1.0;
    let zp = e.clone() + 1.0;
    let o = z.clone() / zp.clone();

    o.backward();

    assert_eq!(e.borrow_data(), 5.828735);
    assert_eq!(
        zp.borrow_grad(),
        z.borrow_data() * (-1.0 / zp.borrow_data().powi(2))
    );
    assert_eq!(z.borrow_grad(), 1.0 / zp.borrow_data());
    assert_eq!(e.borrow_grad(), z.borrow_grad() + zp.borrow_grad());
    assert_eq!(tmp.borrow_grad(), e.borrow_data() * e.borrow_grad());
}

#[test]
fn test_tanh_comp() {
    let mut n = Value::new(0.8814);
    n.label = Some("n".to_string());
    let mut e = (n.clone() * 2.0).exp();
    e.label = Some("e".to_string());
    let mut o = (e.clone() - 1.0) / (e.clone() + 1.0);
    o.label = Some("o".to_string());
    o.backward();

    let n2 = Value::new(0.8814);
    let o2 = n2.tanh();
    o2.backward();

    assert_eq!(o2.borrow_data(), o.borrow_data());
    assert!((n.borrow_grad() - n2.borrow_grad()).abs() < 0.001);
}

#[test]
fn test_middle_double_dep() {
    let mut a = Value::new(1.0);
    a.label = Some("a".to_string());
    let mut b = a.clone() + 1.0;
    b.label = Some("b".to_string());
    let mut c = b.clone() * 3.0;
    c.label = Some("c".to_string());
    let mut d = b.clone() * 4.0;
    d.label = Some("d".to_string());
    let mut e = c.clone() * d.clone();
    e.label = Some("e".to_string());
    e.backward();

    assert_eq!(d.borrow_grad(), 6.0);
    assert_eq!(c.borrow_grad(), 8.0);
    assert_eq!(b.borrow_grad(), 4.0 * 6.0 + 3.0 * 8.0);
    assert_eq!(a.borrow_grad(), 48.0);
}

#[test]
fn test_karpathy_backward_new_tanh() {
    let x1 = Value::new(2.0);
    let x2 = Value::new(0.0);
    let w1 = Value::new(-3.0);
    let w2 = Value::new(1.0);
    let b = Value::new(6.8813735870195432);
    let x1w1 = x1.clone() * w1.clone();
    let x2w2 = x2.clone() * w2.clone();
    let x1w1x2w2 = x1w1.clone() + x2w2.clone();
    let n = x1w1x2w2.clone() + b.clone();

    let e = (n.clone() * 2.0).exp();
    let o = (e.clone() - 1.0) / (e.clone() + 1.0);
    o.backward();

    assert!((o.borrow_data() - 0.7071).abs() < 0.00001);
    assert_eq!(o.borrow_grad(), 1.0);
    dbg!(n.borrow_grad());
    assert!((n.borrow_grad() - 0.5).abs() < 0.0001);
    assert!((x1w1x2w2.borrow_grad() - 0.5).abs() < 0.0001);
    assert!((b.borrow_grad() - 0.5).abs() < 0.0001);
    assert!((x2w2.borrow_grad() - 0.5).abs() < 0.0001);
    assert!((x1w1.borrow_grad() - 0.5).abs() < 0.0001);
    assert!((x1.borrow_grad() - -1.5).abs() < 0.0001);
    assert!((w1.borrow_grad() - 1.0).abs() < 0.0001);
    assert!((x2.borrow_grad() - 0.5).abs() < 0.0001);
    assert!((w2.borrow_grad() - 0.0).abs() < 0.0001);
}
