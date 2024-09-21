pub trait AIModule {
	type Input;
	type Output;
	type LearningCoeff;

	fn forward(&self, input: &Self::Input) -> Self::Output;
	fn backpropagate(&self,
		backward_input: &Self::Output,
		forward_input: &Self::Input,
		forward_output: &Self::Output
	) -> Self::Input;
	fn learn(&mut self,
		backward_input: &Self::Output,
		forward_input: &Self::Input,
		forward_output: &Self::Output,
		learning_rate: Self::LearningCoeff
	);

	fn learn_and_propagate(&mut self,
		backward_input: &Self::Output,
		forward_input: &Self::Input,
		forward_output: &Self::Output,
		learning_rate: Self::LearningCoeff
	) -> Self::Input
	{
		let out = self.backpropagate(backward_input, forward_input, forward_output);
		self.learn(backward_input, forward_input, forward_output, learning_rate);
		out
	}
}
