//! Compile and run quantum programs using a simulator with dense linear algebra algorithms on the GPU.
//!
//! Implementation notes:
//! Since I couldn't get CubeCL to work with a custom Complex type, I decided to transform the complex vectors/matrices into a tensor with 1 more dimension of size 2,
//! which represent the real/imaginary part.
//! For example:
//! - matrix: shape \[n, n\] -> tensor of shape \[n, n, 2\] (real, imaginary)
//! - vector: shape \[n\] -> tensor of shape \[n, 2\] (real, imaginary)

use std::marker::PhantomData;

use crate::{
	AppliedGate, ComplexMatrix,
	backend::{Backend, BackendOperation, LookupTable, Program},
	circuit::Circuit,
	gates,
	state::StateVector,
};

use cubecl::{prelude::*, server::Allocation};
use num::{Complex, complex::ComplexFloat};

/// Turns a slice of f64's into bytes.
fn bytes_from_f64_slice(src: &[f64]) -> Vec<u8> {
	let mut bytes = Vec::with_capacity(src.len() * core::mem::size_of::<f64>());
	for &f in src {
		bytes.extend_from_slice(&f.to_ne_bytes());
	}
	return bytes;
}

/// Turns bytes into a vec of f64's.
fn f64_vec_from_bytes(bytes: &[u8]) -> Vec<f64> {
	let mut out = Vec::with_capacity(bytes.len() / core::mem::size_of::<f64>());
	for chunk in bytes.chunks_exact(core::mem::size_of::<f64>()) {
		let mut arr = [0u8; 8];
		arr.copy_from_slice(chunk);
		out.push(f64::from_ne_bytes(arr));
	}
	return out;
}

/// A [complex matrix](crate::complex_matrix) where the components are linearized and not split into real/imaginary anymore, and therefore able to be transferred to the GPU.
#[derive(Debug, Clone)]
struct ReadyForGPUComplexMatrix {
	/// The components of the matrix, stored in row-major order, with real and imaginary parts of a single complex value contiguous.
	components: Vec<f64>,
}

impl ReadyForGPUComplexMatrix {
	/// Turns a regular complex matrix into one ready for GPU transfer.
	fn from_matrix(mat: &ComplexMatrix) -> Self {
		let n = mat.size_side();
		let mut components = Vec::with_capacity(n * n * 2);
		for r in 0..n {
			for c in 0..n {
				let v = mat[(r, c)];
				components.push(v.re);
				components.push(v.im);
			}
		}
		return Self { components };
	}

	/// Turns the components of the matrix into bytes.
	fn as_bytes(&self) -> Vec<u8> {
		bytes_from_f64_slice(&self.components)
	}
}

/// A state vector where the components are linearized and not split into real/imaginary anymore, and therefore able to be transferred to the GPU.
#[derive(Debug, Clone)]
struct ReadyForGPUStateVector {
	/// The components of the vector, with real and imaginary parts of a single complex value contiguous.
	components: Vec<f64>,
}

impl ReadyForGPUStateVector {
	/// Turns a regular state vector into one ready for GPU transfer.
	fn from_state(vec: &StateVector) -> Self {
		let n = vec.components().len();
		let mut components = Vec::with_capacity(n * 2);
		for i in 0..n {
			let v = vec.components()[i];
			components.push(v.re());
			components.push(v.im());
		}
		return Self { components };
	}

	/// Creates a state vector of only 0+0i's
	fn zero(nb_qubits: usize) -> Self {
		Self {
			components: vec![0.0; nb_qubits * 2],
		}
	}

	/// Turns the components of the vector into bytes.
	fn as_bytes(&self) -> Vec<u8> {
		bytes_from_f64_slice(&self.components)
	}

	/// Creates a vector from the given bytes.
	fn from_bytes(bytes: &[u8]) -> Self {
		Self {
			components: f64_vec_from_bytes(bytes),
		}
	}

	/// Turns a GPU state vector back into one with split real and imaginary parts.
	fn into_cpu_state(&self) -> StateVector {
		let n = self.components.len() / 2;
		let mut comps = Vec::with_capacity(n);
		for i in 0..n {
			let re = self.components[2 * i];
			let im = self.components[2 * i + 1];
			comps.push(Complex::new(re, im));
		}
		StateVector::from_vec(comps)
	}
}

/// GPU kernel for multiplying a complex matrix with a complex vector.
///
/// # Arguments
///
/// * `matrix` - The matrix to multiply by the vector.
/// * `vector_in` - The vector to multiply with the matrix.
/// * `vector_out` - Where the result will be stored.
/// * `size` - The number of complex values present. The matrix should be of shape [size, size, 2], and the vectors of shape [size, 2]
#[cube(launch)]
fn complex_matrix_mul_vector(matrix: &Tensor<f64>, vector_in: &Tensor<f64>, vector_out: &mut Tensor<f64>, size: u32) {
	let idx = ABSOLUTE_POS;

	// Out of bounds.
	if idx >= size {
		terminate!();
	}

	let row = idx as u32;

	// Accumulate the resulting complex number with 2 sums: one for the real part and one for the imaginary part.
	let mut acc_real = 0.0;
	let mut acc_imag = 0.0;

	// Dot product between row and vector.
	for j in 0..size {
		// Load the matrix element at position (i, j).
		let m_ij_idx = (row * size + j) * 2;
		let m_ij_real = matrix[m_ij_idx];
		let m_ij_imag = matrix[m_ij_idx + 1];

		// Load the vector element at position (j).
		let v_j_idx = j * 2;
		let v_j_real = vector_in[v_j_idx];
		let v_j_imag = vector_in[v_j_idx + 1];

		acc_real += m_ij_real * v_j_real - m_ij_imag * v_j_imag;
		acc_imag += m_ij_real * v_j_imag + m_ij_imag * v_j_real;
	}

	// Store the resulting complex number.
	vector_out[row * 2] = acc_real;
	vector_out[row * 2 + 1] = acc_imag;
}

/// Maximum number of threads/groups that can be launched in a single dimension.
const MAX_GROUPS_PER_DIM: u32 = 1024;

/// Splits the iterations amongst GPU compute units.
fn compute_dispatch(total_groups: usize) -> (CubeCount, CubeDim) {
	let total = total_groups as u32;
	let dim = std::cmp::min(total, MAX_GROUPS_PER_DIM);

	// How many cubes we need (linear) when each cube covers `dim` groups
	let cube_count = (total + dim - 1) / dim;

	// Decompose cube_count into up-to-3 components (cx, cy, cz) where each <= MAX_GROUPS_PER_DIM
	let mut remainder = cube_count;
	let mut counts = [1u32; 3];
	for i in 0..3 {
		counts[i] = std::cmp::min(remainder, MAX_GROUPS_PER_DIM);
		if counts[i] == 0 {
			counts[i] = 1;
		}
		remainder = (remainder + counts[i] - 1) / counts[i];
	}
	let cube_count_struct = CubeCount::Static(counts[0], counts[1], counts[2]);
	let cube_dim = CubeDim::new_3d(dim, 1, 1);
	return (cube_count_struct, cube_dim);
}

/// Helper function to multiply a matrix with a vector that can be called on the host.
/// Mostly used for tests, it's not advised to use it since it always creates the buffers and does the transfers.
#[allow(unused)]
fn complex_matrix_mul_vector_host<R: Runtime>(device: &R::Device, matrix: &ComplexMatrix, vec_in: &StateVector) -> StateVector {
	let client = R::client(device);
	let n = matrix.size_side();
	assert_eq!(vec_in.components().len(), n);

	// Pack host data into split f64 layout
	let mat_bytes = &ReadyForGPUComplexMatrix::from_matrix(matrix).as_bytes();
	let vec_bytes = &ReadyForGPUStateVector::from_state(vec_in).as_bytes();

	let shape_mat = &[n, n, 2usize];
	let shape_vec = &[n, 2usize];

	// Create device-side tensors
	let mat_alloc = client.create_tensor(&mat_bytes, shape_mat, core::mem::size_of::<f64>());
	let vec_in_alloc = client.create_tensor(&vec_bytes, shape_vec, core::mem::size_of::<f64>());
	let vec_out_alloc = client.empty_tensor(shape_vec, core::mem::size_of::<f64>());

	let vectorization = 1u8; // Haven't bothered yet to try with SIMD cause I don't know how to approach it for now

	// Compute dispatch that respects device limits
	let groups = n; // 1 output row per ABSOLUTE_POS
	let (cube_count, cube_dim) = compute_dispatch(groups);

	unsafe {
		complex_matrix_mul_vector::launch::<R>(
			&client,
			cube_count,
			cube_dim,
			TensorArg::from_raw_parts_and_size(
				&mat_alloc.handle,
				&mat_alloc.strides,
				shape_mat,
				vectorization,
				core::mem::size_of::<f64>(),
			),
			TensorArg::from_raw_parts_and_size(
				&vec_in_alloc.handle,
				&vec_in_alloc.strides,
				shape_vec,
				vectorization,
				core::mem::size_of::<f64>(),
			),
			TensorArg::from_raw_parts_and_size(
				&vec_out_alloc.handle,
				&vec_out_alloc.strides,
				shape_vec,
				vectorization,
				core::mem::size_of::<f64>(),
			),
			cubecl::frontend::ScalarArg::<u32>::new(n as u32),
		);
	}

	let out_bytes = client.read_one(vec_out_alloc.handle);
	let out_f64 = f64_vec_from_bytes(&out_bytes);

	// Rebuild StateVector from pairwise real/imaginary
	let mut comps = Vec::with_capacity(n);
	for i in 0..n {
		let real = out_f64[i * 2];
		let imag = out_f64[i * 2 + 1];
		comps.push(Complex::new(real, imag));
	}
	return StateVector::from_vec(comps);
}

/// Backend to compile programs into [DenseGPUProgram]'s
pub struct DenseGPUBackend<R: Runtime> {
	/// Stores the type of runtime used
	marker: PhantomData<R>,
}

impl<R: Runtime> DenseGPUBackend<R> {
	/// Creates a new GPU backend.
	pub fn new() -> Self {
		Self { marker: PhantomData }
	}
}

/// A program that runs a quantum algorithm with dense linear algebra calculations on the GPU.
pub struct DenseGPUProgram<R: Runtime> {
	/// The operations of the algorithm, with fundamental matrices being in the GPU-ready format.
	operations: Vec<BackendOperation<ReadyForGPUComplexMatrix>>,

	/// The allocation of a matrix buffer that resides on the GPU.
	/// All the matrices should be of the same size so the buffer will stay the same throughout execution.
	matrix_gpu_allocation: Allocation,

	/// The allocation of the input state vector that resides on the GPU.
	vector_in_gpu_allocation: Allocation,

	/// The allocation of the resulting state vector from a matrix * vector operation that resides on the GPU.
	vector_out_gpu_allocation: Allocation,

	/// Identifier of which matrix is currently present on the GPU.
	/// Used to prevent useless memory transfers in case it is already present.
	matrix_present_on_gpu: Vec<usize>,

	/// Number of qubits used in the circuit.
	nb_qubits: usize,

	/// The GPU client that will do the computations.
	client: ComputeClient<<R as Runtime>::Server>,
}

impl LookupTable<ReadyForGPUComplexMatrix> {
	/// Creates a look-up table of GPU-ready matrices that can be ran in the simulation.
	pub fn from(look_up_table: &Vec<(Vec<(usize, u8)>, AppliedGate)>, nb_qubits: usize) -> Self {
		let mut table = Vec::new();
		for (values, gate) in look_up_table {
			let matrix = gate.into_matrix(nb_qubits);
			let matrix = ReadyForGPUComplexMatrix::from_matrix(&matrix);
			table.push((values.clone(), matrix));
		}
		return Self { table };
	}
}

impl<R: Runtime> Backend<DenseGPUProgram<R>> for DenseGPUBackend<R> {
	fn compile(&self, circuit: &Circuit) -> DenseGPUProgram<R> {
		let nb_qubits = circuit.nb_qubits();
		let dimension = 2_usize.pow(nb_qubits as u32);

		// Fuse matrices found in a row, and for other operations just add them in order.
		let mut operations = vec![];
		let mut current_matrix = ComplexMatrix::identity(dimension);
		for operation in circuit.steps() {
			match operation {
				gates::QuantumOperation::Gate(gate) => {
					let matrix = gate.into_matrix(nb_qubits);
					current_matrix = matrix * &current_matrix;
				}
				gates::QuantumOperation::ClassicalControl { look_up_table } => {
					operations.push(BackendOperation::Matrix(ReadyForGPUComplexMatrix::from_matrix(&current_matrix)));
					current_matrix = ComplexMatrix::identity(dimension);
					operations.push(BackendOperation::ClassicalControl {
						look_up_table: LookupTable::<ReadyForGPUComplexMatrix>::from(&look_up_table, nb_qubits),
					});
				}
				gates::QuantumOperation::Measure(on) => {
					operations.push(BackendOperation::Matrix(ReadyForGPUComplexMatrix::from_matrix(&current_matrix)));
					current_matrix = ComplexMatrix::identity(dimension);
					operations.push(BackendOperation::Measure(*on));
				}
			}
		}
		// Push matrix built if ended with a matrix
		if let Some(gates::QuantumOperation::Gate(_)) = circuit.steps().last() {
			operations.push(BackendOperation::Matrix(ReadyForGPUComplexMatrix::from_matrix(&current_matrix)));
		}

		// Allocate the needed buffers on the GPU
		let matrix_shape = &[dimension, dimension, 2];
		let vector_shape = &[dimension, 2usize];
		let device = Default::default();
		let client = R::client(&device);
		let matrix_gpu_allocation = client.empty_tensor(matrix_shape, core::mem::size_of::<f64>());
		let vector_in_gpu_allocation = client.empty_tensor(vector_shape, core::mem::size_of::<f64>());
		let vector_out_gpu_allocation = client.empty_tensor(vector_shape, core::mem::size_of::<f64>());

		return DenseGPUProgram {
			operations,
			matrix_gpu_allocation,
			vector_in_gpu_allocation,
			vector_out_gpu_allocation,
			matrix_present_on_gpu: vec![],
			nb_qubits,
			client,
		};
	}
}

impl<R: Runtime> DenseGPUProgram<R> {
	/// Writes a matrix to the GPU, only if it's not already present.
	///
	/// # Arguments
	///
	/// * `matrix` - The GPU-ready but CPU-side matrix to transfer.
	/// * `transferred_matrix_idx` - The identifier of the matrix that needs to be transferred.
	/// * `matrix_on_gpu` - The identifier of the matrix that it currently present on the GPU. If it's the same as [transferred_matrix_idx], nothing will be done.
	fn write_matrix_to_gpu_if_needed(
		&mut self, matrix: &ReadyForGPUComplexMatrix, transferred_matrix_idx: &[usize], matrix_on_gpu: Vec<usize>,
	) {
		if transferred_matrix_idx == matrix_on_gpu {
			return;
		}

		let dimension = 2_usize.pow(self.nb_qubits as u32);
		let matrix_shape = &[dimension, dimension, 2];

		// TODO: Verify if cubecl does the extra allocation or knows how to reuse it
		self.matrix_gpu_allocation = self
			.client
			.create_tensor(&matrix.as_bytes(), matrix_shape, core::mem::size_of::<f64>());

		self.matrix_present_on_gpu = transferred_matrix_idx.iter().map(|x| *x).collect();
	}

	/// Writes a vector to the input vector in the GPU.
	///
	/// # Arguments
	///
	/// * `vec` - The CPU-side vector to transfer.
	fn write_vector_in_to_gpu(&mut self, vec: &ReadyForGPUStateVector) {
		let dimension = 2_usize.pow(self.nb_qubits as u32);
		let vector_shape = &[dimension, 2];

		// TODO: Also verify if cubecl does the extra allocation or knows how to reuse it
		self.vector_in_gpu_allocation = self
			.client
			.create_tensor(&vec.as_bytes(), vector_shape, core::mem::size_of::<f64>());
	}

	/// Reads the output vector back to the CPU.
	fn read_vector_out_from_gpu(&mut self) -> ReadyForGPUStateVector {
		let out_bytes = self.client.read_one(self.vector_out_gpu_allocation.handle.clone());
		ReadyForGPUStateVector::from_bytes(&out_bytes)
	}

	/// Performs a matrix * vector multiplication on the GPU, that is called from the host
	///
	/// # Arguments
	/// * `matrix` - The GPU-ready but CPU-side matrix to multiply with.
	/// * `matrix_idx` - The identifier of that matrix.
	/// * `state` - The vector that will get multiplied by the matrix.

	fn matrix_times_state(
		&mut self, matrix: &ReadyForGPUComplexMatrix, matrix_idx: Vec<usize>, state: &ReadyForGPUStateVector,
	) -> ReadyForGPUStateVector {
		let matrix_present_on_gpu = self.matrix_present_on_gpu.clone();
		self.write_matrix_to_gpu_if_needed(matrix, &matrix_idx, matrix_present_on_gpu);
		self.write_vector_in_to_gpu(state);

		let dimension = 2_usize.pow(self.nb_qubits as u32);

		let vectorization = 1u8; // Haven't bothered yet to try with SIMD cause I don't know how to approach it for now

		// Compute dispatch that respects device limits
		let groups = dimension; // 1 output row per ABSOLUTE_POS
		let (cube_count, cube_dim) = compute_dispatch(groups);

		let shape_mat = &[dimension, dimension, 2usize];
		let shape_vec = &[dimension, 2usize];

		unsafe {
			complex_matrix_mul_vector::launch::<R>(
				&self.client,
				cube_count,
				cube_dim,
				TensorArg::from_raw_parts_and_size(
					&self.matrix_gpu_allocation.handle,
					&self.matrix_gpu_allocation.strides,
					shape_mat,
					vectorization,
					core::mem::size_of::<f64>(),
				),
				TensorArg::from_raw_parts_and_size(
					&self.vector_in_gpu_allocation.handle,
					&self.vector_in_gpu_allocation.strides,
					shape_vec,
					vectorization,
					core::mem::size_of::<f64>(),
				),
				TensorArg::from_raw_parts_and_size(
					&self.vector_out_gpu_allocation.handle,
					&self.vector_out_gpu_allocation.strides,
					shape_vec,
					vectorization,
					core::mem::size_of::<f64>(),
				),
				cubecl::frontend::ScalarArg::<u32>::new(dimension as u32),
			);
		}

		return self.read_vector_out_from_gpu();
	}

	/// Runs the simulation, branches into sub-simulations in cases of measure or classical control operations.
	///
	/// # Arguments
	///
	/// * `steps` - The remaining operations to be simulated.
	/// * `matrix_on_gpu` - The identifier of the matrix present on the GPU
	/// * `current_step` - At which step the program is in. When it move forwards it adds 1 to the last element, and when it branches, it adds a new element.
	/// * `state` - The current state of the input state vector.
	/// * `bit_values` - The measured value (0/1) of each qubit in the case where they got measured.
	/// * `nb_qubits` - The number of qubits the program runs on.
	fn run_and_branch(
		&mut self, steps: &[BackendOperation<ReadyForGPUComplexMatrix>], matrix_on_gpu: Vec<usize>, current_step: Vec<usize>,
		state: &ReadyForGPUStateVector, bit_values: &[(usize, u8)], nb_qubits: usize,
	) -> ReadyForGPUStateVector {
		// If end of program is reached, return the calculated state
		if steps.is_empty() {
			return state.clone();
		}

		match &steps[0] {
			BackendOperation::Matrix(matrix) => {
				// Move one step forward in the current branch
				let mut new_current_step = current_step.clone();
				let last_step = new_current_step.last_mut().unwrap();
				*last_step += 1;

				let new_state = self.matrix_times_state(matrix, new_current_step.clone(), state);
				return self.run_and_branch(
					&steps[1..],
					self.matrix_present_on_gpu.clone(),
					current_step,
					&new_state,
					bit_values,
					nb_qubits,
				);
			}
			BackendOperation::Measure(on_qubit) => {
				// Compute the outcomes if the qubit ends up being measured as 0 or 1 and how likely each one are
				let mut outcomes = Vec::new();
				for possible_qubit_value in [0u8, 1] {
					// Loop over all possible combinations of outcomes, and keep only the ones where the desired qubit has the desired value
					// And sum their probabilities (amplitudes squared) to know how likely it is to measure it in that value.
					let mut state_if_measured = ReadyForGPUStateVector::zero(nb_qubits);
					let mut probability_to_measure = 0.0;
					for possible_outcome in 0..state.components.len() {
						let qubit_value = possible_outcome >> (nb_qubits - 1 - on_qubit) & 1;
						if qubit_value as u8 == possible_qubit_value {
							state_if_measured.components[2 * possible_outcome] =
								state.components[possible_outcome];
							state_if_measured.components[2 * possible_outcome + 1] =
								state.components[possible_outcome + 1];

							let as_complex = Complex::from(state_if_measured.components[2 * possible_outcome])
								+ state_if_measured.components[2 * possible_outcome + 1] * Complex::i();
							probability_to_measure += as_complex.norm_sqr();
						}
					}

					// If it can happen, normalize the state and add it to the possible outcomes
					if probability_to_measure > 0.0 {
						for amplitude in state_if_measured.components.iter_mut() {
							*amplitude /= probability_to_measure;
						}
						outcomes.push((possible_qubit_value, probability_to_measure, state_if_measured));
					}
				}

				// Run simulations on the rest of the circuit assuming either state, and aggregate the results at the end
				let mut combined_outcomes = ReadyForGPUStateVector::zero(nb_qubits);
				// let remaining_operations = &steps[1..];
				for (bit_value, probability, branched_state) in outcomes {
					let mut bit_values: Vec<_> = bit_values.iter().map(|x| *x).collect();
					bit_values.push((*on_qubit, bit_value));

					let mut new_current_steps = current_step.clone();
					new_current_steps.push(bit_value as usize);
					let final_state = self.run_and_branch(
						steps,
						matrix_on_gpu.clone(),
						new_current_steps,
						&branched_state,
						&bit_values,
						nb_qubits,
					);

					for i in 0..final_state.components.len() {
						combined_outcomes.components[i] += probability * final_state.components[i];
					}
				}
				return combined_outcomes;
			}

			BackendOperation::ClassicalControl { look_up_table } => {
				// Find gate to apply and remove the placeholder one, then go on with the rest of the computation
				let (idx, gate_to_apply) = look_up_table
					.get_enumerate(bit_values)
					.expect("No matching condition found in classical control");

				let mut new_steps: Vec<_> = steps.iter().map(|x| x.clone()).collect();
				let mut new_current_steps = current_step.clone();
				new_current_steps.push(idx);
				new_steps[0] = BackendOperation::Matrix(gate_to_apply.clone());

				return self.run_and_branch(&new_steps, matrix_on_gpu, new_current_steps, state, bit_values, nb_qubits);
			}
		}
	}
}

impl<R: Runtime> Program for DenseGPUProgram<R> {
	fn run(&mut self, state: &StateVector) -> StateVector {
		let gpu_state = ReadyForGPUStateVector::from_state(state);
		let operations = self.operations.clone();
		let gpu_final_state = self.run_and_branch(&operations, vec![], vec![0], &gpu_state, &[], self.nb_qubits);
		return gpu_final_state.into_cpu_state();
	}
}

#[cfg(test)]
mod tests {
	use crate::state::Ket;

	use super::*;

	const NB_RANDOM_TESTS: usize = 20;

	/// Checks at a 2*id * vector = 2 * vector, computation done on the GPU.
	#[test]
	fn gpu_matrix_vector_2_times_identity_doubles_state() {
		let nb_qubits = 5usize;
		let dimension = 2usize.pow(nb_qubits as u32);

		// Id * 2 -> Should multiply the input by 2
		let mat = ComplexMatrix::identity(dimension) * Complex::from(2.0);

		for _ in 0..NB_RANDOM_TESTS {
			let base = Ket::random(dimension);
			let state = StateVector::from_ket(base.clone());

			let device = Default::default();
			let result = complex_matrix_mul_vector_host::<cubecl::wgpu::WgpuRuntime>(&device, &mat, &state);

			let expected_state =
				StateVector::from_ket(Ket::from_components(base.components().iter().map(|z| z * 2.0).collect()));
			assert!(result.approx_eq(&expected_state, 1e-6));
		}
	}
}
