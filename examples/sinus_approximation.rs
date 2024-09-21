use deep_learning::*;
use deep_learning::dense::*;
use deep_learning::activation::*;
use deep_learning::vector::*;

fn cos(x: f32) -> f32 {
    x.cos()
}

fn cos_derivate(x: f32, _y: f32) -> f32 {
    -(x.sin())
}

fn main() {
    let cst_2pi = 2. * std::f32::consts::PI;

	let mut d = Dense::<1, 10000>::rand(2.);
    let mut a = ActivationLayer1D::<10000>::new(cos, cos_derivate);

	let learning_rate = 0.25;
    for _ in 0 .. 10000 {
		let x = (rand::random::<f32>() - 0.5) * 2. * cst_2pi;
		let desired_output = Vector::new(x.sin());
        sequential_train!(learning_rate, Vector::from([x]), desired_output, d, a);
    }
	
	for i in 0 .. 10000 {
		println!("{}; Weight: {}, Bias: {}", i,
			d.weights()[i][0], d.bias()[i]);
	}

	let mut sum_score = Vector::new(0.);
	let tests_count = 100;
	for _ in 0 .. tests_count {
		let x = rand::random::<f32>() * cst_2pi;
		let desired_output = Vector::new(x.sin());
		let real_output = sequential_forward!(Vector::from([x]), d, a);
		let score = real_output - desired_output;
		sum_score += score * score;
	}

	for i in 0 .. 10000 {
		println!("{}; Average score: {}, Weight: {}, Bias: {}", i,
			sum_score[i] / tests_count as f32,
			d.weights()[i][0], d.bias()[i]);
	}
}
