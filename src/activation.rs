use crate::ai_trait::*;
use crate::vector::*;
use crate::matrix::*;

pub struct ActivationLayer1D<const N: usize> {
	f: fn(f32) -> f32,
	df: fn(f32, f32) -> f32
}

impl<const N: usize>
ActivationLayer1D<N>
{
	pub fn new(f: fn(f32) -> f32, df: fn(f32, f32) -> f32) -> Self
	{ Self{f: f, df: df} }
}

impl<const N: usize>
AIModule for ActivationLayer1D<N>
{
	type Input = Vector<N>;
	type Output = Vector<N>;
	type LearningCoeff = f32;

	fn forward(&self, input: &Self::Input) -> Self::Output
	{
		let mut output = Vector::new(0.);
		for i in 0 .. N {
			output[i] = (self.f)(input[i]);
		}
		return output;
	}

	fn backpropagate(&self,
		backward_input: &Self::Output,
		forward_input: &Self::Input,
		forward_output: &Self::Output
	) -> Self::Input
	{
		let mut input = Vector::new(0.);
		for i in 0 .. N {
			input[i] = (self.df)(forward_input[i], forward_output[i])
                * backward_input[i];
		}
		return input;
	}

	fn learn(&mut self,
		_backward_input: &Self::Output,
		_forward_input: &Self::Input,
		_forward_output: &Self::Output,
		_learning_rate: Self::LearningCoeff
	)
	{ }
}


pub struct ActivationLayer2D<const N: usize, const M: usize, const L: usize> {
	f: fn(f32) -> f32,
	df: fn(f32, f32) -> f32
}

impl<const N: usize, const M: usize, const L: usize>
ActivationLayer2D<N, M, L>
{
	pub fn new(f: fn(f32) -> f32, df: fn(f32, f32) -> f32) -> Self
	{ Self{f: f, df: df} }
}

impl<const N: usize, const M: usize, const L: usize>
AIModule for ActivationLayer2D<N, M, L>
{
	type Input = [Matrix<N, M>; L];
	type Output = [Matrix<N, M>; L];
	type LearningCoeff = f32;

	fn forward(&self, input: &Self::Input) -> Self::Output
	{
		let mut output = [Matrix::new(0.); L];
		for l in 0 .. L {
			for j in 0 .. M {
				for i in 0 .. N {
					output[l][j][i] = (self.f)(input[l][j][i]);
				}
			}
		}
		return output;
	}

	fn backpropagate(&self,
		backward_input: &Self::Output,
		forward_input: &Self::Input,
		forward_output: &Self::Output
	) -> Self::Input
	{
		let mut output = [Matrix::new(0.); L];
		for l in 0 .. L {
			for j in 0 .. M {
				for i in 0 .. N {
					output[l][j][i] = (self.df)(forward_input[l][j][i], forward_output[l][j][i])
                        * backward_input[l][j][i];
				}
			}
		}
		return output;
	}

	fn learn(&mut self,
		_backward_input: &Self::Output,
		_forward_input: &Self::Input,
		_forward_output: &Self::Output,
		_learning_rate: Self::LearningCoeff
	)
	{ }
}
