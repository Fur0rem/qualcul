use num::{Complex, One, Zero};

use crate::ComplexMatrix;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Ket {
	components: Vec<Complex<f64>>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Bra {
	components: Vec<Complex<f64>>,
}

impl Ket {
	pub fn base(x: usize, nb_dimensions: usize) -> Self {
		let mut components = vec![Complex::zero(); nb_dimensions];
		components[x] = Complex::one();
		return Self { components };
	}

	pub fn dimension(&self) -> usize {
		self.components.len()
	}

	pub fn add_and_normalize(a: &Ket, b: &Ket) -> Self {
		assert!(a.dimension() == b.dimension());
		let dimension = a.dimension();

		let norm = (a
			.components
			.iter()
			.zip(b.components.iter())
			.map(|(x, y)| {
				let sum = *x + *y;
				sum.norm_sqr()
			})
			.sum::<f64>())
		.sqrt();

		let mut result = Vec::with_capacity(dimension);
		for component_idx in 0..dimension {
			result.push(Complex::from(1.0 / norm) * (a.components[component_idx] + b.components[component_idx]));
		}

		return Self { components: result };
	}

	pub fn sub_and_normalize(a: &Ket, b: &Ket) -> Self {
		assert!(a.dimension() == b.dimension());
		let dimension = a.dimension();

		let norm = (a
			.components
			.iter()
			.zip(b.components.iter())
			.map(|(x, y)| {
				let diff = *x - *y;
				diff.norm_sqr()
			})
			.sum::<f64>())
		.sqrt();

		let mut result = Vec::with_capacity(dimension);
		for component_idx in 0..dimension {
			result.push(Complex::from(1.0 / norm) * (a.components[component_idx] - b.components[component_idx]));
		}

		return Self { components: result };
	}

	pub fn into_bra(&self) -> Bra {
		Bra {
			components: self.components.iter().map(|z| z.conj()).collect(),
		}
	}

	pub fn ket_plus() -> Self {
		return Self::add_and_normalize(&Self::base(0b0, 2), &Self::base(0b1, 2));
	}

	pub fn ket_minus() -> Self {
		return Self::sub_and_normalize(&Self::base(0b0, 2), &Self::base(0b1, 2));
	}

	pub fn bell_phi_plus() -> Self {
		return Self::add_and_normalize(&Self::base(0b00, 4), &Self::base(0b11, 4));
	}

	pub fn bell_psi_plus() -> Self {
		return Self::add_and_normalize(&Self::base(0b01, 4), &Self::base(0b10, 4));
	}

	pub fn bell_phi_minus() -> Self {
		return Self::sub_and_normalize(&Self::base(0b00, 4), &Self::base(0b11, 4));
	}

	pub fn bell_psi_minus() -> Self {
		return Self::sub_and_normalize(&Self::base(0b01, 4), &Self::base(0b10, 4));
	}

	pub fn projector(&self) -> ComplexMatrix {
		let self_bra = self.into_bra();
		return self * self_bra;
	}

	pub fn components(&self) -> &[Complex<f64>] {
		&self.components
	}
}

impl Bra {
	pub fn base(x: usize, nb_dimensions: usize) -> Self {
		let mut components = vec![Complex::zero(); nb_dimensions];
		components[x] = Complex::one();
		return Self { components };
	}

	pub fn into_ket(&self) -> Ket {
		Ket {
			components: self.components.iter().map(|z| z.conj()).collect(),
		}
	}

	pub fn dimension(&self) -> usize {
		self.components.len()
	}
}

// Projectors
impl std::ops::Mul<Bra> for &Ket {
	type Output = ComplexMatrix;

	fn mul(self, rhs: Bra) -> Self::Output {
		assert!(self.dimension() == rhs.dimension());

		let dimension = self.dimension();
		let mut result = ComplexMatrix::zero(dimension);

		for i in 0..dimension {
			for j in 0..dimension {
				result[(i, j)] += self.components[i] * rhs.components[j];
			}
		}

		return result;
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_projector() {
		let projector_0 = Ket::base(0b0, 2).projector();
		let expected_projector = ComplexMatrix::from(&vec![
			vec![Complex::from(1.0), Complex::from(0.0)],
			vec![Complex::from(0.0), Complex::from(0.0)],
		]);
		assert!(projector_0.approx_eq(&expected_projector, 1e-6));

		let projector_1 = Ket::base(0b1, 2).projector();
		let expected_projector = ComplexMatrix::from(&vec![
			vec![Complex::from(0.0), Complex::from(0.0)],
			vec![Complex::from(0.0), Complex::from(1.0)],
		]);
		assert!(projector_1.approx_eq(&expected_projector, 1e-6));
	}
}
