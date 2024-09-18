use std::io::Error;
use std::fs;
use std::iter::Iterator;
use std::path::Path;
use crate::deep_learning::vector::*;
use crate::deep_learning::matrix::*;

#[allow(unused_macros)]
macro_rules! read2_little {
	($data:ident, $start:expr) => {
		($data[$start]     as u16) << 8 |
		($data[$start + 1] as u16) << 0
	};
}

macro_rules! read4_little {
	($data:ident, $start:expr) => {
		(read2_little!($data, $start + 2) as u32) << 0 |
		(read2_little!($data, $start)     as u32) << 16
	};
}

#[derive(Clone, Copy)]
pub struct MnistEntry<const N: usize, const M: usize> {
	pub data: Matrix<N, M>,
	pub class: u32
}

pub struct Mnist<const N: usize, const M: usize> {
	entries: Vec<MnistEntry<N, M>>
}

pub struct MnistIterator<const N: usize, const M: usize> {
	entries: Vec<MnistEntry<N, M>>,
	entry_to_send: usize
}

impl<const N: usize, const M: usize> Mnist<N, M> {
	pub fn new<P: AsRef<Path>>(label_path: P, images_path: P) -> Result<Self, Error> {
		let images = fs::read(images_path);
		if let Err(e) = images {
			return Err(e);
		}

		let labels = fs::read(label_path);
		if let Err(e) = labels {
			return Err(e);
		}

		let images = images.unwrap();
		let labels = labels.unwrap();

		let (nimages, width, height) = (
			read4_little!(images, 4) as usize,
			read4_little!(images, 8) as usize,
			read4_little!(images, 12) as usize
		);

		assert_eq!(width, N);
		assert_eq!(height, M);

		let mut entries = Vec::with_capacity(nimages);

		for i in 0 .. nimages {
			let mut values = [Vector::new(0.); M];

			for y in 0 .. M {
				for x in 0 .. N {
					values[y][x] = images[(i * M + y) * N + x + 16] as f32 / 255.;
				}	
			}

			entries.push(MnistEntry{
				data: Matrix::from(values),
				class: labels[i + 8] as u32
			});
		}

		return Ok(Mnist{entries: entries});
	}

	pub fn nb_entries(&self) -> usize
	{ self.entries.len() }

	pub fn iter(&self) -> MnistIterator<N, M>
	{
		MnistIterator {
			entries: self.entries.clone(),
			entry_to_send: 0
		}
	}
}

impl<const N: usize, const M: usize>
Iterator for MnistIterator<N, M> {
	type Item = MnistEntry<N, M>;

	fn next(&mut self) -> Option<Self::Item>
	{
		let tmp = self.entry_to_send;

		self.entry_to_send += 1;

		if self.entry_to_send >= self.entries.len() {
			self.entry_to_send = 0;
			None
		} else {
			Some(self.entries[tmp])
		}
	}
}