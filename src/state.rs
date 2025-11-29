use num::{Complex, One, Zero, complex::ComplexFloat};
use rand::Rng;

use crate::ComplexMatrix;

#[derive(Clone, PartialEq, Default)]
pub struct Ket {
	components: Vec<Complex<f64>>,
}

impl std::fmt::Debug for Ket {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		let max_bits = (self.components.len() as f64).log2().ceil() as usize;
		write!(f, "Ket ")?;
		let mut first = true;
		for (idx, amplitude) in self.components().iter().enumerate() {
			let binary_state = format!("{:0width$b}", idx, width = max_bits);
			if *amplitude != Complex::zero() {
				if first {
					write!(f, "{}+{}i |{}>", amplitude.re(), amplitude.im(), binary_state)?;
					first = false;
				} else {
					writeln!(f, "")?;
					write!(f, " + {}+{}i |{}>", amplitude.re(), amplitude.im(), binary_state)?;
				}
			}
		}
		return Ok(());
	}
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Bra {
	components: Vec<Complex<f64>>,
}

impl Ket {
	pub fn from_components(components: Vec<Complex<f64>>) -> Self {
		return Self { components };
	}

	pub fn base(x: usize, nb_dimensions: usize) -> Self {
		let mut components = vec![Complex::zero(); nb_dimensions];
		components[x] = Complex::one();
		return Self { components };
	}

	pub fn dimension(&self) -> usize {
		self.components.len()
	}

	pub fn add_multiple_and_normalize(kets: &[Ket], coefficients: &[Complex<f64>]) -> Self {
		assert!(kets.len() == coefficients.len());
		let dimension = kets[0].dimension();

		let norm = (0..kets.len())
			.map(|i| {
				let ket = &kets[i];
				let coefficient = coefficients[i];
				ket.components.iter().map(|c| (coefficient * c).norm_sqr()).sum::<f64>()
			})
			.sum::<f64>()
			.sqrt();

		let mut result = vec![Complex::zero(); dimension];
		for i in 0..kets.len() {
			let ket = &kets[i];
			let coefficient = coefficients[i];
			for component_idx in 0..dimension {
				result[component_idx] += (coefficient / Complex::from(norm)) * ket.components[component_idx];
			}
		}

		return Self { components: result };
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

	pub fn kronecker_product(&self, other: &Ket) -> Ket {
		let dim_a = self.dimension();
		let dim_b = other.dimension();
		let mut components = vec![Complex::zero(); dim_a * dim_b];

		for i in 0..dim_a {
			for j in 0..dim_b {
				components[i * dim_b + j] = self.components[i] * other.components[j];
			}
		}

		return Ket { components };
	}

	pub fn approx_eq(&self, rhs: &Ket, epsilon: f64) -> bool {
		if self.components().len() != rhs.components().len() {
			return false;
		}
		for i in 0..self.components().len() {
			let difference = self.components()[i] - rhs.components()[i];
			if difference.re().abs() > epsilon || difference.im().abs() > epsilon {
				return false;
			}
		}
		return true;
	}

	pub fn random(nb_dimensions: usize) -> Self {
		let mut rng = rand::rng();
		let mut components = Vec::with_capacity(nb_dimensions);
		for _ in 0..nb_dimensions {
			let re: f64 = rng.random_range(-1.0..1.0);
			let im: f64 = rng.random_range(-1.0..1.0);
			components.push(Complex::new(re, im));
		}
		// Normalize
		let norm = components.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
		for c in components.iter_mut() {
			*c /= Complex::from(norm);
		}
		return Self { components };
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
