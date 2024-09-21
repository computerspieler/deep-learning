use std::ops;

#[derive(Debug)]
pub struct Vector<const N: usize>([f32; N]);

impl<const N: usize> Vector<N> {
	pub fn new(value: f32) -> Self {
		Vector([value; N])
	}

	pub fn from(values: [f32; N]) -> Self {
		Vector(values)
	}

	pub fn rand(jitterness: f32) -> Self {
		let mut output = [0.; N];
		for i in 0 .. N {
			output[i] = (2. * rand::random::<f32>() - 1.) * jitterness;
		}
		return Vector(output);
	}

	pub fn dim(&self) -> usize
	{ N }
	
	pub fn dot(&self, rhs: &Vector<N>) -> f32 {
		let mut output = 0.;
		
		for i in 0 .. N {
			output += self.0[i] * rhs.0[i];
		}
		
		return output;
	}

	pub fn len2(&self) -> f32 {
		self.dot(&self)
	}

	pub fn reverse(&mut self) {
		self.0.reverse();
	}
}

impl<const N: usize> Clone for Vector<N> {
	fn clone(&self) -> Self {
		Self(self.0.clone())	
	}
}

impl<const N: usize> Copy for Vector<N>
{ }

impl<const N: usize> ops::Index<usize> for Vector<N>
{
	type Output = f32;

	fn index(&self, index: usize) -> &f32
	{
		assert!(index < N);
		return &self.0[index];
	}
}

impl<const N: usize> ops::IndexMut<usize> for Vector<N>
{
	fn index_mut(&mut self, index: usize) -> &mut f32
	{
		assert!(index < N);
		return &mut self.0[index];
	}
}

impl<const N: usize> ops::Add<Vector<N>> for Vector<N> {
	type Output = Vector<N>;

	fn add(self, rhs: Vector<N>) -> Vector<N>
	{
		let mut output = Vector::new(0.);
		for i in 0 .. N {
			output.0[i] = self.0[i] + rhs.0[i];
		}

		output
	}
}

impl<const N: usize> ops::AddAssign<Vector<N>> for Vector<N> {
	fn add_assign(&mut self, rhs: Vector<N>)
	{
		for i in 0 .. N {
			self.0[i] += rhs.0[i];
		}
	}
}

impl<const N: usize> ops::Sub<Vector<N>> for Vector<N> {
	type Output = Vector<N>;

	fn sub(self, rhs: Vector<N>) -> Self
	{
		let mut output = Vector::new(0.);
		for i in 0 .. N {
			output.0[i] = self.0[i] - rhs.0[i];
		}

		output
	}
}

impl<const N: usize> ops::SubAssign<Vector<N>> for Vector<N> {
	fn sub_assign(&mut self, rhs: Vector<N>)
	{
		for i in 0 .. N {
			self.0[i] -= rhs.0[i];
		}
	}
}

impl<const N: usize> ops::Mul<f32> for Vector<N> {
	type Output = Vector<N>;

	fn mul(self, rhs: f32) -> Self
	{
		let mut output = self.clone();
		for i in 0 .. N {
			output.0[i] *= rhs;
		}

		output
	}
}

impl<const N: usize> ops::Mul<Vector<N>> for Vector<N> {
	type Output = Vector<N>;

	fn mul(self, rhs: Vector<N>) -> Self
	{
		let mut output = self.clone();
		for i in 0 .. N {
			output.0[i] *= rhs[i];
		}

		output
	}
}
