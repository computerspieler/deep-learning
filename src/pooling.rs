use crate::ai_trait::*;
use crate::matrix::*;

pub enum PoolingLayer2D<const N: usize, const M: usize, const L: usize, const POOL_SIZE: usize> {
	MaxPooling,
	AvgPooling	
}

impl<const N: usize, const M: usize, const L: usize, const POOL_SIZE: usize>
AIModule for PoolingLayer2D<N, M, L, POOL_SIZE>
where [(); N / POOL_SIZE]: Sized,
      [(); M / POOL_SIZE]: Sized
{
	type Input = [Matrix<N, M>; L];
	type Output = [Matrix<{N / POOL_SIZE}, {M / POOL_SIZE}>; L];
	type LearningCoeff = f32;

	fn forward(&self, input: &Self::Input) -> Self::Output
	{
		let mut output = [Matrix::new(0.); L];
		for l in 0 .. L {
			for j in 0 .. M / POOL_SIZE {
				for i in 0 .. N / POOL_SIZE {
					match *self {
					Self::MaxPooling => {
						let mut res: f32 = input[l][j * POOL_SIZE][i * POOL_SIZE];
						
						for y in (j*POOL_SIZE) .. ((j+1) * POOL_SIZE).min(M) {
							for x in (i*POOL_SIZE) .. ((i+1) * POOL_SIZE).min(N) {
								res = res.max(input[l][y][x]);
							}
						}

						output[l][j][i] = res;
					}
					Self::AvgPooling => {
						let mut res = 0.;

						for y in (j*POOL_SIZE) .. ((j+1) * POOL_SIZE).min(M) {
							for x in (i*POOL_SIZE) .. ((i+1) * POOL_SIZE).min(N) {
								res += input[l][y][x];
							}
						}
						
						output[l][j][i] = res / ((POOL_SIZE * POOL_SIZE) as f32);
					}
					}
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
			for j in 0 .. M / POOL_SIZE {
				for i in 0 .. N / POOL_SIZE {
					match *self {
						Self::MaxPooling => {
							let mut pos_max = (i*POOL_SIZE, j*POOL_SIZE);
							for y in (j*POOL_SIZE) .. ((j+1) * POOL_SIZE).min(M) {
								for x in (i*POOL_SIZE) .. ((i+1) * POOL_SIZE).min(N) {
									if forward_output[l][j][i] == forward_input[l][y][x] {
										pos_max = (x, y);
									}
								}
							}

							output[l][pos_max.1][pos_max.0] = backward_input[l][j][i];
						}
						Self::AvgPooling => {
							let coeff = backward_input[l][j][i] / ((POOL_SIZE * POOL_SIZE) as f32);
							for x in (i*POOL_SIZE) .. ((i+1) * POOL_SIZE).min(N) {
								for y in (j*POOL_SIZE) .. ((j+1) * POOL_SIZE).min(M) {
									output[l][y][x] = coeff;
								}
							}
						}
					}
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

