use std::ops;
use crate::vector::*;

#[derive(Debug)]
pub struct Matrix<const N: usize, const M: usize>([Vector<N>; M]);

impl<const N: usize, const M: usize> Matrix<N, M> {
	pub fn new(value: f32) -> Self {
		Self([Vector::new(value); M])
	}
	
	pub fn from(values: [Vector<N>; M]) -> Self {
		Self(values)
	}

	pub fn rand(jitterness: f32) -> Self {
		let mut output = [Vector::rand(jitterness); M];
		for i in 1 .. M {
			output[i] = Vector::rand(jitterness);
		}

		Self(output)
	}

	pub fn t(&self) -> Matrix<M, N> {
		let mut output = Matrix::new(0.);

		for i in 0 .. N {
			for j in 0 .. M {
				output[i][j] = self[j][i]
			}	
		}

		return output;
	}
	
	pub fn dot(self, rhs: &Vector<N>) -> Vector<M>
	{
		let mut output = Vector::new(0.);
		
		for i in 0 .. M {
			output[i] += (*rhs).dot(&self[i]);
		}
		
		output
	}

	pub fn rotate180(&self) -> Matrix<N, M>
	{
		let mut output = Matrix::from(self.0);
		output.0.reverse();
		
		for j in 0 .. M {
			output[j].reverse();
		}

		return output;
	}
	
	pub fn conv<const PAD_N: usize, const PAD_M: usize,
		const IN_N: usize, const IN_M: usize,
		const OUT_N: usize, const OUT_M: usize>
	(&self, input: Matrix<IN_N, IN_M>) -> Matrix<OUT_N, OUT_M>
	{
		assert_eq!(OUT_N, IN_N + 2*PAD_N - N + 1);
		assert_eq!(OUT_M, IN_M + 2*PAD_M - M + 1);
		
		let mut output = Matrix::new(0.);

		for j in 0 .. OUT_M {
			let y0: i32 = j as i32 - PAD_M as i32;
			for i in 0 .. OUT_N {
				let x0: i32 = i as i32 - PAD_N as i32;
			
				let mut res = 0.;

				for y in 0.max(-y0) .. M as i32 {
					let yi = y + y0;
					if yi >= IN_M as i32 {
						break;
					}

					for x in 0.max(-x0) .. N as i32 {
						let xi = x + x0;
						if xi >= IN_N as i32 {
							break;
						}

						res += self[y as usize][x as usize] *
							input[yi as usize][xi as usize];
					}
				}

				output[j][i] = res;
			}	
		}

		return output;
	}
}

impl<const N: usize, const M: usize> Clone for Matrix<N, M> {
	fn clone(&self) -> Self {
		Self(self.0.clone())	
	}
}

impl<const N: usize, const M: usize> Copy for Matrix<N, M> 
{ }

impl<const N: usize, const M: usize>
ops::Index<usize> for Matrix<N, M>
{
	type Output = Vector<N>;

	fn index(&self, index: usize) -> &Vector<N>
	{
		assert!(index < M);
		return &self.0[index];
	}
}

impl<const N: usize, const M: usize>
ops::IndexMut<usize> for Matrix<N, M>
{
	fn index_mut(&mut self, index: usize) -> &mut Vector<N>
	{
		assert!(index < M);
		return &mut self.0[index];
	}
}

impl<const N: usize, const M: usize>
ops::Add<Matrix<N, M>> for Matrix<N, M> {
	type Output = Matrix<N, M>;

	fn add(self, rhs: Self) -> Self
	{
		let mut output = Matrix::new(0.);
		for i in 0 .. N {
			output.0[i] = self.0[i] + rhs.0[i];
		}

		output
	}
}

impl<const N: usize, const M: usize>
ops::AddAssign<Matrix<N, M>> for Matrix<N, M> {
	fn add_assign(&mut self, rhs: Matrix<N, M>) {
		for i in 0 .. N {
			self.0[i] += rhs.0[i];
		}
	}
}

impl<const N: usize, const M: usize>
ops::Sub<Matrix<N, M>> for Matrix<N, M> {
	type Output = Matrix<N, M>;

	fn sub(self, rhs: Self) -> Self
	{
		let mut output = Matrix::new(0.);
		for i in 0 .. N {
			output.0[i] = self.0[i] - rhs.0[i];
		}

		output
	}
}

impl<const N: usize, const M: usize>
ops::Mul<f32> for Matrix<N, M> {
	type Output = Matrix<N, M>;

	fn mul(self, rhs: f32) -> Self
	{
		let mut output = Matrix::new(0.);
		for i in 0 .. N {
			output.0[i] = self.0[i] * rhs;
		}

		output
	}
}
