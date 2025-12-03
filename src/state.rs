//! States qubits can be in.

use num::{
	Complex, One, Zero,
	complex::{Complex64, ComplexFloat},
};
use rand::Rng;

use crate::ComplexMatrix;

/// A state in ket notation.
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
					write!(f, "{}+{}i |{}⟩", amplitude.re(), amplitude.im(), binary_state)?;
					first = false;
				}
				else {
					writeln!(f, "")?;
					write!(f, " + {}+{}i |{}⟩", amplitude.re(), amplitude.im(), binary_state)?;
				}
			}
		}
		return Ok(());
	}
}

/// A state in bra notation.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Bra {
	components: Vec<Complex<f64>>,
}

impl Ket {
	/// Creates a ket state from a given vector of complex numbers.
	pub fn from_components(components: Vec<Complex<f64>>) -> Self {
		return Self { components };
	}

	/// Creates a ket state from the canonical base.
	///
	/// # Arguments
	///
	/// * `x` - The axis of the base.
	/// * `nb_dimensions` - The number of dimensions of the state, should be equal to 2 to the power of the number of qubits.
	///
	/// # Examples
	/// ```
	/// use qualcul::state::Ket;
	/// let state = Ket::base(0b00100, 32); // Represents |00100⟩ , or |4⟩ depending on which notation you prefer.
	/// ```
	pub fn base(x: usize, nb_dimensions: usize) -> Self {
		let mut components = vec![Complex::zero(); nb_dimensions];
		components[x] = Complex::one();
		return Self { components };
	}

	/// Gets the dimension of the state.
	pub fn dimension(&self) -> usize {
		self.components.len()
	}

	/// Does the operation c1*K1 + c2*K2 + ... + cn*Kn and normalizes the result to put it back in the unit hyper-sphere.
	///
	/// # Arguments
	///
	/// * `kets` - The states to add.
	/// * `coefficients` - Each complex coefficient to multiply each state with.
	///
	/// # Panics
	///
	/// This function panics if kets and coefficients don't have the same size.
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

	/// Adds two states and normalizes the result to put it back in the unit hyper-sphere.
	///
	/// # Panics
	///
	/// This function panics if the 2 states aren't of the same dimension
	pub fn add_and_normalize(lhs: &Ket, rhs: &Ket) -> Self {
		assert!(lhs.dimension() == rhs.dimension());
		let dimension = lhs.dimension();

		let norm = (lhs
			.components
			.iter()
			.zip(rhs.components.iter())
			.map(|(x, y)| {
				let sum = *x + *y;
				sum.norm_sqr()
			})
			.sum::<f64>())
		.sqrt();

		let mut result = Vec::with_capacity(dimension);
		for component_idx in 0..dimension {
			result.push(Complex::from(1.0 / norm) * (lhs.components[component_idx] + rhs.components[component_idx]));
		}

		return Self { components: result };
	}

	/// Subtracts two states and normalizes the result to put it back in the unit hyper-sphere.
	///
	/// # Panics
	///
	/// This function panics if the 2 states aren't of the same dimension
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

	/// Turns the state into its equivalent bra state, i.e. its adjoint.
	pub fn into_bra(&self) -> Bra {
		Bra {
			components: self.components.iter().map(|z| z.conj()).collect(),
		}
	}

	/// Creates a |+⟩ state.
	pub fn ket_plus() -> Self {
		return Self::add_and_normalize(&Self::base(0b0, 2), &Self::base(0b1, 2));
	}

	/// Creates a |-⟩ state.
	pub fn ket_minus() -> Self {
		return Self::sub_and_normalize(&Self::base(0b0, 2), &Self::base(0b1, 2));
	}

	/// Creates a |Φ+⟩ state in Bell's base
	pub fn bell_phi_plus() -> Self {
		return Self::add_and_normalize(&Self::base(0b00, 4), &Self::base(0b11, 4));
	}

	/// Creates a |Ψ-⟩ state in Bell's base
	pub fn bell_psi_plus() -> Self {
		return Self::add_and_normalize(&Self::base(0b01, 4), &Self::base(0b10, 4));
	}

	/// Creates a |Φ-⟩ state in Bell's base
	pub fn bell_phi_minus() -> Self {
		return Self::sub_and_normalize(&Self::base(0b00, 4), &Self::base(0b11, 4));
	}

	/// Creates a |Ψ-⟩ state in Bell's base
	pub fn bell_psi_minus() -> Self {
		return Self::sub_and_normalize(&Self::base(0b01, 4), &Self::base(0b10, 4));
	}

	/// Computes the projector of the state.
	pub fn projector(&self) -> ComplexMatrix {
		let self_bra = self.into_bra();
		return self * self_bra;
	}

	/// Gets the components of the state.
	pub fn components(&self) -> &[Complex<f64>] {
		&self.components
	}

	/// Computes the kronecker product of a state with `rhs`.
	pub fn kronecker_product(&self, rhs: &Ket) -> Ket {
		let dim_a = self.dimension();
		let dim_b = rhs.dimension();
		let mut components = vec![Complex::zero(); dim_a * dim_b];

		for i in 0..dim_a {
			for j in 0..dim_b {
				components[i * dim_b + j] = self.components[i] * rhs.components[j];
			}
		}

		return Ket { components };
	}

	/// Checks whether 2 states are approximately equal given an error threshold.
	///
	/// # Arguments
	///
	/// * `rhs` - The other state to compare with.
	/// * `epsilon` - The margin of error tolerated between 2 elements.
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

	/// Creates a random and valid (normalized) state.
	///
	/// # Arguments
	///
	/// * `dimension` - The dimension of the state.
	pub fn random(dimension: usize) -> Self {
		let mut rng = rand::rng();
		let mut components = Vec::with_capacity(dimension);
		for _ in 0..dimension {
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
	/// Creates a bra state from the canonical base.
	///
	/// # Arguments
	///
	/// * `x` - The axis of the base.
	/// * `nb_dimensions` - The number of dimensions of the state, should be equal to 2 to the power of the number of qubits.
	///
	/// # Examples
	/// ```
	/// use qualcul::state::Bra;
	/// let state = Bra::base(0b00100, 32); // Represents ⟨00100| , or ⟨4| depending on which notation you prefer.
	/// ```
	pub fn base(x: usize, nb_dimensions: usize) -> Self {
		let mut components = vec![Complex::zero(); nb_dimensions];
		components[x] = Complex::one();
		return Self { components };
	}

	/// Turns the state into its equivalent ket state, i.e. its adjoint.
	pub fn into_ket(&self) -> Ket {
		Ket {
			components: self.components.iter().map(|z| z.conj()).collect(),
		}
	}

	/// Gets the dimension of the state.
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

/// The state vector of the qubits used for simulations in a circuit.
#[derive(Clone, PartialEq, Default)]
pub struct StateVector(pub(crate) Vec<Complex<f64>>);

impl std::fmt::Debug for StateVector {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		let max_bits = (self.0.len() as f64).log2().ceil() as usize;
		writeln!(f, "StateVector (")?;
		for (idx, amplitude) in self.0.iter().enumerate() {
			let binary_state = format!("{:0width$b}", idx, width = max_bits);
			writeln!(f, "    |{}⟩: {}+{}i", binary_state, amplitude.re(), amplitude.im())?;
		}
		write!(f, ")")?;
		return Ok(());
	}
}

impl StateVector {
	/// Turns a complex vector into a state vector.
	pub fn from_vec(state: Vec<Complex<f64>>) -> Self {
		StateVector(state)
	}

	/// Turns a ket state into a state vector.
	pub fn from_ket(state: Ket) -> Self {
		Self::from_vec(state.components)
	}

	/// Creates a state vector from the individual states in which each qubit is.
	pub fn from_qubits(qubit_states: &[&Ket]) -> Self {
		let mut result = qubit_states[0].clone();
		for qubit_state in &qubit_states[1..] {
			result = result.kronecker_product(qubit_state);
		}
		Self::from_vec(result.components().to_vec())
	}

	/// Checks whether 2 states are approximately equal given an error threshold.
	///
	/// # Arguments
	///
	/// * `rhs` - The other state to compare with.
	/// * `epsilon` - The margin of error tolerated between 2 elements.
	pub fn approx_eq(&self, rhs: &StateVector, epsilon: f64) -> bool {
		if self.0.len() != rhs.0.len() {
			return false;
		}
		for i in 0..self.0.len() {
			let difference = self.0[i] - rhs.0[i];
			if difference.re().abs() > epsilon || difference.im().abs() > epsilon {
				return false;
			}
		}
		return true;
	}

	/// Computes the possible outcomes when measuring all the qubits at once, and gives their probability of each state happening.
	/// Doesn't include states with 0 chance of happening.
	// FIXME: I should probably return something other that a Ket because it doesn't make sense to return a quantum state.
	pub fn possible_outcomes(&self) -> Vec<(Ket, f64)> {
		let mut outcomes = Vec::new();
		for (idx, amplitude) in self.0.iter().enumerate() {
			let probability = amplitude.norm_sqr();
			if probability > 0.0 {
				outcomes.push((Ket::base(idx, self.0.len()), probability));
			}
		}
		return outcomes;
	}

	/// Extracts the state in which a single qubit is.
	///
	/// # Arguments
	///
	/// * `qubit` - The index of the qubit to extract the state of.
	pub fn extract_state_of_single_qubit(&self, qubit: usize) -> Ket {
		let dim = self.0.len();
		let mut state0 = Complex64::zero();
		let mut state1 = Complex64::zero();
		for i in 0..dim {
			let qubit_value = (i >> (dim.trailing_zeros() - 1 - qubit as u32)) & 1;
			if qubit_value == 0 {
				state0 += self.0[i];
			}
			else {
				state1 += self.0[i];
			}
		}
		let norm = (state0.norm_sqr() + state1.norm_sqr()).sqrt();
		if norm == 0.0 {
			return Ket::base(0, 2);
		}
		return Ket::from_components(vec![state0 / Complex::from(norm), state1 / Complex::from(norm)]);
	}

	/// Computes the most likely outcome when measuring the qubits.
	/// Returns only 1 of them if there are multiple that are as likely as each other.
	pub fn most_likely_outcome(&self) -> usize {
		self.components()
			.iter()
			.enumerate()
			.max_by(|a, b| a.1.norm_sqr().partial_cmp(&b.1.norm_sqr()).unwrap())
			.map(|(i, _)| i)
			.unwrap()
	}

	/// Gets the components of the state.
	pub fn components(&self) -> &[Complex<f64>] {
		&self.0
	}
}

impl std::ops::Mul<StateVector> for ComplexMatrix {
	type Output = StateVector;
	fn mul(self, rhs: StateVector) -> Self::Output {
		let mut result = StateVector(vec![Complex::from(0.0); rhs.0.len()]);
		for i in 0..self.size_side() {
			for j in 0..self.size_side() {
				result.0[i] += self[(i, j)] * rhs.0[j];
			}
		}
		return result;
	}
}

impl std::ops::Mul<&StateVector> for ComplexMatrix {
	type Output = StateVector;
	fn mul(self, rhs: &StateVector) -> Self::Output {
		let mut result = StateVector(vec![Complex::from(0.0); rhs.0.len()]);
		for i in 0..self.size_side() {
			for j in 0..self.size_side() {
				result.0[i] += self[(i, j)] * rhs.0[j];
			}
		}
		return result;
	}
}

impl std::ops::Mul<&StateVector> for &ComplexMatrix {
	type Output = StateVector;
	fn mul(self, rhs: &StateVector) -> Self::Output {
		let mut result = StateVector(vec![Complex::from(0.0); rhs.0.len()]);
		for i in 0..self.size_side() {
			for j in 0..self.size_side() {
				result.0[i] += self[(i, j)] * rhs.0[j];
			}
		}
		return result;
	}
}

impl std::ops::Index<usize> for StateVector {
	type Output = Complex<f64>;

	fn index(&self, index: usize) -> &Self::Output {
		return &self.0[index];
	}
}

impl std::ops::IndexMut<usize> for StateVector {
	fn index_mut(&mut self, index: usize) -> &mut Self::Output {
		return &mut self.0[index];
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
