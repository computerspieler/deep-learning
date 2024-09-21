#[macro_export]
macro_rules! sequential_forward {
	($input:expr, $($layer:ident),*) => {{
		let _x = $input;
		$(
			let _x = ($layer).forward(&_x);
		)*
		_x
	}};
}

#[macro_export]
macro_rules! sequential_train {
	($learning_rate: expr, $input: expr, $expected: expr, $layer:ident, $($next_layers:ident),*) => {{
		let _x = $input;
		let _fx = ($layer).forward(&_x);
		let (dx, score) = sequential_train!($learning_rate, _fx, $expected, $($next_layers),*);
		(($layer).learn_and_propagate(&dx, &_x, &_fx, $learning_rate), score)
	}};

	($learning_rate: expr, $input: expr, $expected: expr, $layer: ident) => {{
		let _x = $input;
		let _fx = ($layer).forward(&_x);
		let diff = _fx - ($expected);
		let score = _fx.len2() / (_fx.dim() as f32);
		let dx = diff * (2. / (_fx.dim() as f32));
		(($layer).learn_and_propagate(&dx, &_x, &_fx, $learning_rate), score)
	}};
}
/*
fn learn_and_propagate(&mut self,
		backward_input: &Self::Output,
		forward_input: &Self::Input,
		forward_output: &Self::Output,
		learning_rate: Self::LearningCoeff
	) 
*/