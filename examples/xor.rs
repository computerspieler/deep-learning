use deep_learning::*;
use deep_learning::dense::*;
use deep_learning::activation::*;
use deep_learning::vector::*;

fn sigmoid(x: f32) -> f32 {
	1. / (1. + (-x).exp())
}

fn sigmoid_derivate(_x: f32, y: f32) -> f32 {
	y * (1. - y)
}

fn main() {
	let mut d1 = Dense::<2, 2>::rand(1.);
    let mut a1 = ActivationLayer1D::<2>::new(sigmoid, sigmoid_derivate);
	let mut d2 = Dense::<2, 1>::rand(1.);
    let mut a2 = ActivationLayer1D::<1>::new(sigmoid, sigmoid_derivate);

	let dataset = [
		(Vector::from([0., 0.]), Vector::new(0.)),
		(Vector::from([1., 0.]), Vector::new(1.)),
		(Vector::from([0., 1.]), Vector::new(1.)),
		(Vector::from([1., 1.]), Vector::new(0.))
	];

	let learning_rate = 0.25;
    for _ in 0 .. 10000 {
		for (input, desired_output) in dataset {
			sequential_train!(learning_rate, input, desired_output,
				d1, a1, d2, a2);
		}
    }
	
	for (input, _) in dataset {
		let output = sequential_forward!(input,
			d1, a1, d2, a2);
		println!("{} ^ {} = {}",
			input[0], input[1], output[0]);
	}
}
