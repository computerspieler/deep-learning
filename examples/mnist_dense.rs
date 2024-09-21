#![feature(generic_const_exprs)]

use deep_learning::*;
use deep_learning::dense::*;
use deep_learning::activation::*;
use deep_learning::vector::*;
use deep_learning::flatten2d::*;
use deep_learning::dataset::mnist::*;

fn sigmoid(x: f32) -> f32 {
	1. / (1. + (-x).exp())
}

fn sigmoid_derivate(_x: f32, y: f32) -> f32 {
	y * (1. - y)
}

fn main() {	
	println!("Load the training dataset");
	let train_dt = Mnist::<28, 28>::new("mnist_data/train-labels-idx1-ubyte", "mnist_data/train-images-idx3-ubyte");
	if let Err(e) = train_dt {
		dbg!(e);
		return;
	}
	let train_dt = train_dt.unwrap();

	println!("Load the test dataset");
	let test_dt = Mnist::<28, 28>::new("mnist_data/t10k-labels-idx1-ubyte", "mnist_data/t10k-images-idx3-ubyte");
	if let Err(e) = test_dt {
		dbg!(e);
		return;
	}
	let test_dt = test_dt.unwrap();

	println!("Create the model");
	let mut f1 = Flatten2D::<28, 28, 1>::new();
	let mut d1 = Dense::<{28 * 28}, 120>::rand(1.);
	let mut ad1 = ActivationLayer1D::<120>::new(sigmoid, sigmoid_derivate);
	let mut d2 = Dense::<120, 120>::rand(1.);
	let mut ad2 = ActivationLayer1D::<120>::new(sigmoid, sigmoid_derivate);
	let mut d3 = Dense::<120, 120>::rand(1.);
	let mut ad3 = ActivationLayer1D::<120>::new(sigmoid, sigmoid_derivate);
	let mut d4 = Dense::<120, 10>::rand(1.);
	let mut ad4 = ActivationLayer1D::<10>::new(sigmoid, sigmoid_derivate);
	
	let learning_rate = 0.25;
	for _ in 0 .. 7 {
		println!("Training time !");
		let mut desired_output = Vector::from([0.; 10]);
		let mut done = 0;
		let mut sum_score = 0.;
		for ex in train_dt.iter() {
			print!("{} / {}; Avg score: {}\r", done, train_dt.nb_entries(),
				sum_score / done as f32);

			desired_output[ex.class as usize] = 1.;

			let (_, score) = sequential_train!(learning_rate, [ex.data; 1], desired_output,
				f1,
				d1, ad1,
				d2, ad2,
				d3, ad3,
				d4, ad4
			);
			//dbg!(dx);
			sum_score += score;

			desired_output[ex.class as usize] = 0.;
			done += 1;
		}

		println!("\nTesting time !");
		let mut score = 0;
		for ex in test_dt.iter() {
			let xf = sequential_forward!([ex.data; 1],
				f1,
				d1, ad1,
				d2, ad2,
				d3, ad3,
				d4, ad4
			);

			let mut max_id = 0;
			for i in 1 .. 10 {
				if xf[i] >= xf[max_id] {
					max_id = i;
				}
			}

			if max_id == (ex.class as usize) {
				score += 1;
			}
		}
		println!("Score: {} / {}", score, test_dt.nb_entries());
	}
}
