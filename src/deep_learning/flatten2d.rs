use crate::deep_learning::ai_trait::*;
use crate::deep_learning::vector::*;
use crate::deep_learning::matrix::*;

pub struct Flatten2D<const N: usize, const M: usize, const L: usize>
{ }

impl<const N: usize, const M: usize, const L: usize>
Flatten2D<N, M, L>
{
	pub fn new() -> Self
	{ Self{} }
}

impl<const N: usize, const M: usize, const L: usize>
AIModule for Flatten2D<N, M, L>
where Vector<{N * M * L}>: Sized
{
	type Input = [Matrix<N, M>; L];
	type Output = Vector<{N * M * L}>;
	type LearningCoeff = f32;

	fn forward(&self, input: &Self::Input) -> Self::Output
	{
		let mut output = Vector::new(0.);
		for l in 0 .. L {
			for j in 0 .. M {
				for i in 0 .. N {
					output[(l * M + j) * N + i] = input[l][j][i];
				}
			}
		}
		return output;
	}

	fn backpropagate(&self,
		backward_input: &Self::Output,
		_forward_input: &Self::Input,
		_forward_output: &Self::Output
	) -> Self::Input
	{
		let mut output = [Matrix::new(0.); L];
		for l in 0 .. L {
			for j in 0 .. M {
				for i in 0 .. N {
					output[l][j][i] = backward_input[(l * M + j) * N + i];
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
