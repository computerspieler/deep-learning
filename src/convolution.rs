use crate::matrix::*;
use crate::ai_trait::*;

pub struct Conv2D
<const IN_N: usize, const IN_M: usize, const IN_DEPTH: usize,
	const KERNEL_N: usize, const KERNEL_M: usize, 
	const FILTER_COUNT: usize,
	const PAD_N: usize, const PAD_M: usize>
{
	kernels: [[Matrix<KERNEL_N, KERNEL_M>; IN_DEPTH]; FILTER_COUNT]
}

impl<const IN_N: usize, const IN_M: usize, const IN_DEPTH: usize,
	const KERNEL_N: usize, const KERNEL_M: usize, 
	const FILTER_COUNT: usize,
	const PAD_N: usize, const PAD_M: usize>
Conv2D<IN_N, IN_M, IN_DEPTH, KERNEL_N, KERNEL_M, FILTER_COUNT, PAD_N, PAD_M>
{
	pub fn rand(jitterness: f32) -> Self
	{
		let mut kernels =
			[[Matrix::new(0.); IN_DEPTH]; FILTER_COUNT];
		for i in 0 .. FILTER_COUNT {
			for j in 0 .. IN_DEPTH {
				kernels[i][j] = Matrix::rand(jitterness);
			}
		}
		Self{
			kernels: kernels
		}
	}
}

impl<const IN_N: usize, const IN_M: usize, const IN_DEPTH: usize,
	const KERNEL_N: usize, const KERNEL_M: usize, 
	const FILTER_COUNT: usize,
	const PAD_N: usize, const PAD_M: usize>
AIModule for Conv2D<IN_N, IN_M, IN_DEPTH, KERNEL_N, KERNEL_M, FILTER_COUNT, PAD_N, PAD_M>
where [(); IN_N + 2*PAD_N - KERNEL_N + 1]: Sized,
	  [(); IN_M + 2*PAD_M - KERNEL_M + 1]: Sized,
	  [(); KERNEL_N - 1 - PAD_N]: Sized,
	  [(); KERNEL_M - 1 - PAD_M]: Sized,
{
	type Input = [Matrix<IN_N, IN_M>; IN_DEPTH];
	type Output = [Matrix<
		{IN_N + 2*PAD_N - KERNEL_N + 1},
		{IN_M + 2*PAD_M - KERNEL_M + 1}
	>; FILTER_COUNT];
	type LearningCoeff = f32;

	fn forward(&self, input: &Self::Input) -> Self::Output
	{
		let mut output = [Matrix::new(0.); FILTER_COUNT];

		for filter_idx in 0 .. FILTER_COUNT {
			for input_idx in 0 .. IN_DEPTH {
				output[filter_idx] +=
					self.kernels[filter_idx][input_idx].conv::<PAD_N, PAD_M, _, _, _, _>
					(input[input_idx]);
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
		let mut output: [Matrix<IN_N, IN_M>; IN_DEPTH] = [Matrix::new(0.); IN_DEPTH];

		for input_idx in 0 .. IN_DEPTH {
			for filter_idx in 0 .. FILTER_COUNT {
				output[input_idx] +=
					self.kernels[filter_idx][input_idx].rotate180().conv::<
						{KERNEL_N - 1 - PAD_N},
						{KERNEL_M - 1 - PAD_M},
						{IN_N + 2*PAD_N - KERNEL_N + 1},
						{IN_M + 2*PAD_M - KERNEL_M + 1},
						IN_N, IN_M
					>(backward_input[filter_idx]);
			}
		}
		
		return output;
	}

	fn learn(&mut self,
		backward_input: &Self::Output,
		forward_input: &Self::Input,
		_forward_output: &Self::Output,
		learning_rate: Self::LearningCoeff
	)
	{
		for filter_idx in 0 .. FILTER_COUNT {
			for input_idx in 0 .. IN_DEPTH {
				let tmp = backward_input[filter_idx].conv::<
					PAD_N, PAD_M,
					IN_N, IN_M,
					KERNEL_N, KERNEL_M
				>(forward_input[input_idx]);
				self.kernels[filter_idx][input_idx] += tmp * (-learning_rate);
			}
		}
	}
}
