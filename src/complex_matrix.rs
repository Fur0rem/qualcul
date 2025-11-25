use num::{Complex, complex::ComplexFloat};

#[derive(Clone, PartialEq, Default)]
pub struct ComplexMatrix {
	values: Vec<Complex<f64>>,
	size_side: usize,
}

impl std::fmt::Debug for ComplexMatrix {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		writeln!(f)?;
		for i in 0..self.size_side {
			for j in 0..self.size_side {
				write!(f, "{}+{}i ", self[(i, j)].re(), self[(i, j)].im())?;
			}
			writeln!(f)?;
		}
		return Ok(());
	}
}

impl ComplexMatrix {
	pub fn zero(size_side: usize) -> Self {
		Self {
			values: vec![Complex::new(0.0, 0.0); size_side * size_side],
			size_side,
		}
	}

	pub fn from(values: &Vec<Vec<Complex<f64>>>) -> Self {
		let size_side = values.len();
		for row in values {
			assert!(row.len() == size_side);
		}

		let mut result = ComplexMatrix::zero(size_side);
		for i in 0..size_side {
			for j in 0..size_side {
				result[(i, j)] = values[i][j];
			}
		}

		return result;
	}

	pub fn size_side(&self) -> usize {
		self.size_side
	}

	pub fn identity(size_side: usize) -> Self {
		let mut result = ComplexMatrix::zero(size_side);

		for i in 0..size_side {
			result[(i, i)] = Complex::from(1.0);
		}

		return result;
	}
}

impl std::ops::Index<(usize, usize)> for ComplexMatrix {
	type Output = Complex<f64>;
	fn index(&self, index: (usize, usize)) -> &Self::Output {
		return &self.values[index.1 * self.size_side + index.0];
	}
}

impl std::ops::IndexMut<(usize, usize)> for ComplexMatrix {
	fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
		return &mut self.values[index.1 * self.size_side + index.0];
	}
}

// TODO: One add operation for with/without references
impl std::ops::Add<ComplexMatrix> for ComplexMatrix {
	type Output = ComplexMatrix;

	fn add(self, rhs: ComplexMatrix) -> Self::Output {
		assert!(self.size_side() == rhs.size_side());
		let mut result = ComplexMatrix::zero(self.size_side());
		for i in 0..self.size_side() {
			for j in 0..self.size_side() {
				result[(i, j)] = self[(i, j)] + rhs[(i, j)];
			}
		}
		return result;
	}
}

impl<'a, 'b> std::ops::Add<&'b ComplexMatrix> for &'a ComplexMatrix {
	type Output = ComplexMatrix;

	fn add(self, rhs: &'b ComplexMatrix) -> Self::Output {
		assert!(self.size_side() == rhs.size_side());
		let mut result = ComplexMatrix::zero(self.size_side());
		for i in 0..self.size_side() {
			for j in 0..self.size_side() {
				result[(i, j)] = self[(i, j)] + rhs[(i, j)];
			}
		}
		return result;
	}
}

impl<'a, 'b> std::ops::Mul<&'b ComplexMatrix> for &'a ComplexMatrix {
	type Output = ComplexMatrix;

	fn mul(self, rhs: &'b ComplexMatrix) -> Self::Output {
		assert!(rhs.size_side == self.size_side);

		let mut result = ComplexMatrix::zero(self.size_side);
		for i in 0..self.size_side {
			for j in 0..self.size_side {
				for k in 0..self.size_side {
					result[(i, j)] += self[(i, k)] * rhs[(k, j)];
				}
			}
		}
		return result;
	}
}

impl std::ops::MulAssign<Complex<f64>> for ComplexMatrix {
	fn mul_assign(&mut self, rhs: Complex<f64>) {
		for i in 0..self.size_side {
			for j in 0..self.size_side {
				self[(i, j)] *= rhs;
			}
		}
	}
}

impl std::ops::Mul<Complex<f64>> for ComplexMatrix {
	type Output = ComplexMatrix;

	fn mul(self, rhs: Complex<f64>) -> Self::Output {
		let mut result = ComplexMatrix::zero(self.size_side);
		for i in 0..self.size_side {
			for j in 0..self.size_side {
				result[(i, j)] = rhs * self[(i, j)];
			}
		}
		return result;
	}
}

impl ComplexMatrix {
	pub fn transpose(&self) -> Self {
		let mut result = Self::zero(self.size_side);

		for i in 0..self.size_side {
			for j in 0..self.size_side {
				result[(i, j)] = self[(j, i)];
			}
		}

		return result;
	}

	pub fn conjugate(&self) -> Self {
		let mut result = Self::zero(self.size_side);

		for i in 0..self.size_side {
			for j in 0..self.size_side {
				result[(i, j)] = self[(j, i)].conj();
			}
		}

		return result;
	}

	pub fn adjoint(&self) -> Self {
		return self.conjugate().transpose();
	}

	pub fn kronecker_product(&self, rhs: &Self) -> Self {
		let mut result = ComplexMatrix::zero(self.size_side() * rhs.size_side());
		for i in 0..self.size_side() {
			for j in 0..self.size_side() {
				for k in 0..rhs.size_side() {
					for l in 0..rhs.size_side() {
						result[(i * rhs.size_side() + k, j * rhs.size_side() + l)] = self[(i, j)] * rhs[(k, l)];
					}
				}
			}
		}
		return result;
	}

	pub fn approx_eq(&self, rhs: &Self, epsilon: f64) -> bool {
		if self.size_side() != rhs.size_side() {
			return false;
		}

		for i in 0..self.size_side {
			for j in 0..self.size_side {
				let difference = self[(i, j)] - rhs[(i, j)];
				if difference.re() > epsilon || difference.im() > epsilon {
					return false;
				}
			}
		}
		return true;
	}
}
