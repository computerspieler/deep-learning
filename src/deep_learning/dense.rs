use crate::deep_learning::vector::*;
use crate::deep_learning::matrix::*;
use crate::deep_learning::ai_trait::*;

pub struct Dense<const INPUT_DIM: usize, const OUTPUT_DIM: usize> {
	weights: Matrix<INPUT_DIM, OUTPUT_DIM>,
	bias: Vector<OUTPUT_DIM>
}

impl <const INPUT_DIM: usize, const OUTPUT_DIM: usize>
Dense<INPUT_DIM, OUTPUT_DIM>
{
	pub fn rand(jitterness: f32) -> Self {
		Self {
			weights: Matrix::rand(jitterness),
			bias: Vector::rand(jitterness)
		}
	}
}

impl <const INPUT_DIM: usize, const OUTPUT_DIM: usize>
AIModule for Dense<INPUT_DIM, OUTPUT_DIM> {
	type Input = Vector<INPUT_DIM>;
	type Output = Vector<OUTPUT_DIM>;
	type LearningCoeff = f32;

	fn forward(&self, input: &Self::Input) -> Self::Output
	{ self.weights.dot(input) + self.bias }

	fn backpropagate(&self,
		backward_input: &Self::Output,
		_forward_input: &Self::Input,
		_forward_output: &Self::Output
	) -> Self::Input
	{ self.weights.t().dot(backward_input) }

	fn learn(&mut self,
		backward_input: &Self::Output,
		forward_input: &Self::Input,
		_forward_output: &Self::Output,
		learning_rate: Self::LearningCoeff
	)
	{
		for i in 0 .. OUTPUT_DIM {
			let x = (*backward_input)[i] * learning_rate;
			self.weights[i] -= *forward_input * x;
			self.bias[i] -= x;
		}

	}
}