// Implementation notes:
// Since I couldn't get CubeCL to work with a custom Complex type, I decided to transform the complex vectors/matrices into a tensor with 1 more dimension of size 2,
// which represent the real/imaginary part.
// For example:
// - matrix: shape [n, n] -> tensor of shape [n, n, 2] (re, im)
// - vector: shape [n] -> tensor of shape [n, 2] (re, im)

use std::marker::PhantomData;

use crate::{
	ComplexMatrix, GateOperation,
	backend::{Backend, BackendOperation, LookupTable, Program},
	circuit::{Circuit, StateVector},
	gates,
};

use cubecl::{prelude::*, server::Allocation};
use num::{Complex, complex::ComplexFloat};

fn bytes_from_f64_slice(src: &[f64]) -> Vec<u8> {
	let mut bytes = Vec::with_capacity(src.len() * core::mem::size_of::<f64>());
	for &f in src {
		bytes.extend_from_slice(&f.to_ne_bytes());
	}
	return bytes;
}

fn f64_vec_from_bytes(bytes: &[u8]) -> Vec<f64> {
	let mut out = Vec::with_capacity(bytes.len() / core::mem::size_of::<f64>());
	for chunk in bytes.chunks_exact(core::mem::size_of::<f64>()) {
		let mut arr = [0u8; 8];
		arr.copy_from_slice(chunk);
		out.push(f64::from_ne_bytes(arr));
	}
	return out;
}

#[derive(Debug, Clone)]
struct GPUComplexMatrix {
	components: Vec<f64>,
}

impl GPUComplexMatrix {
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

	fn as_bytes(&self) -> Vec<u8> {
		bytes_from_f64_slice(&self.components)
	}
}

#[derive(Debug, Clone)]
struct GPUStateVector {
	components: Vec<f64>,
}

impl GPUStateVector {
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

	fn zero(nb_qubits: usize) -> Self {
		Self {
			components: vec![0.0; nb_qubits * 2],
		}
	}

	fn as_bytes(&self) -> Vec<u8> {
		bytes_from_f64_slice(&self.components)
	}

	fn from_bytes(bytes: &[u8]) -> Self {
		Self {
			components: f64_vec_from_bytes(bytes),
		}
	}

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

#[cube(launch)]
fn complex_matrix_mul_vector(matrix: &Tensor<f64>, vector_in: &Tensor<f64>, vector_out: &mut Tensor<f64>, size: u32) {
	let idx = ABSOLUTE_POS;
	if idx >= size {
		terminate!();
	}

	let row = idx as u32;
	let mut acc_real: f64 = 0.0;
	let mut acc_imag: f64 = 0.0;

	// Dot product between row and vector
	for j in 0..size {
		// Load m_ij
		let m_ij_idx = (row * size + j) * 2;
		let m_ij_real = matrix[m_ij_idx];
		let m_ij_imag = matrix[m_ij_idx + 1];

		// Load v_j
		let v_j_idx = j * 2;
		let v_j_real = vector_in[v_j_idx];
		let v_j_imag = vector_in[v_j_idx + 1];

		acc_real += m_ij_real * v_j_real - m_ij_imag * v_j_imag;
		acc_imag += m_ij_real * v_j_imag + m_ij_imag * v_j_real;
	}

	vector_out[row * 2] = acc_real;
	vector_out[row * 2 + 1] = acc_imag;
}

const MAX_GROUPS_PER_DIM: u32 = 1024;

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

pub fn complex_matrix_mul_vector_host<R: Runtime>(device: &R::Device, matrix: &ComplexMatrix, vec_in: &StateVector) -> StateVector {
	let client = R::client(device);
	let n = matrix.size_side();
	assert_eq!(vec_in.components().len(), n);

	// Pack host data into split f64 layout
	let mat_bytes = &GPUComplexMatrix::from_matrix(matrix).as_bytes();
	let vec_bytes = &GPUStateVector::from_state(vec_in).as_bytes();

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

pub struct DenseGPUBackend<R: Runtime> {
	marker: PhantomData<R>,
}

impl<R: Runtime> DenseGPUBackend<R> {
	pub fn new() -> Self {
		Self { marker: PhantomData }
	}
}

pub struct DenseGPUProgram<R: Runtime> {
	operations: Vec<BackendOperation<GPUComplexMatrix>>,
	matrix_gpu_allocation: Allocation,
	vector_in_gpu_allocation: Allocation,
	vector_out_gpu_allocation: Allocation,
	matrix_present_on_gpu: Vec<usize>,
	nb_qubits: usize,
	client: ComputeClient<<R as Runtime>::Server>,
}

impl LookupTable<GPUComplexMatrix> {
	pub fn from(look_up_table: &Vec<(Vec<(usize, u8)>, GateOperation)>, nb_qubits: usize) -> Self {
		let mut table = Vec::new();
		for (values, gate) in look_up_table {
			let matrix = gates::as_matrix(&gate, nb_qubits);
			let matrix = GPUComplexMatrix::from_matrix(&matrix);
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
					let matrix = gates::as_matrix(&gate, nb_qubits);
					current_matrix = matrix * &current_matrix;
				}
				gates::QuantumOperation::ClassicalControl { look_up_table } => {
					operations.push(BackendOperation::Matrix(GPUComplexMatrix::from_matrix(&current_matrix)));
					current_matrix = ComplexMatrix::identity(dimension);
					operations.push(BackendOperation::ClassicalControl {
						look_up_table: LookupTable::<GPUComplexMatrix>::from(&look_up_table, nb_qubits),
					});
				}
				gates::QuantumOperation::Measure(on) => {
					operations.push(BackendOperation::Matrix(GPUComplexMatrix::from_matrix(&current_matrix)));
					current_matrix = ComplexMatrix::identity(dimension);
					operations.push(BackendOperation::Measure(*on));
				}
			}
		}
		// Push matrix built if ended with a matrix
		if let Some(gates::QuantumOperation::Gate(_)) = circuit.steps().last() {
			operations.push(BackendOperation::Matrix(GPUComplexMatrix::from_matrix(&current_matrix)));
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
	// TODO: verify if cubecl does the extra allocation or knows how to reuse it
	fn write_matrix_to_gpu_if_needed(&mut self, matrix: &GPUComplexMatrix, current_matrix_idx: &[usize], matrix_on_gpu: Vec<usize>) {
		if current_matrix_idx == matrix_on_gpu {
			return;
		}

		let dimension = 2_usize.pow(self.nb_qubits as u32);
		let matrix_shape = &[dimension, dimension, 2];
		self.matrix_gpu_allocation = self
			.client
			.create_tensor(&matrix.as_bytes(), matrix_shape, core::mem::size_of::<f64>());

		self.matrix_present_on_gpu = current_matrix_idx.iter().map(|x| *x).collect();
	}

	fn write_vector_in_to_gpu(&mut self, vec: &GPUStateVector) {
		let dimension = 2_usize.pow(self.nb_qubits as u32);
		let vector_shape = &[dimension, 2];
		self.vector_in_gpu_allocation = self
			.client
			.create_tensor(&vec.as_bytes(), vector_shape, core::mem::size_of::<f64>());
	}

	fn read_vector_out_from_gpu(&mut self) -> GPUStateVector {
		let out_bytes = self.client.read_one(self.vector_out_gpu_allocation.handle.clone());
		GPUStateVector::from_bytes(&out_bytes)
	}

	fn matrix_times_state(&mut self, matrix: &GPUComplexMatrix, matrix_idx: Vec<usize>, state: &GPUStateVector) -> GPUStateVector {
		let matrix_present_on_gpu = self.matrix_present_on_gpu.clone();
		self.write_matrix_to_gpu_if_needed(matrix, &matrix_idx, matrix_present_on_gpu);
		self.write_vector_in_to_gpu(state);

		let n = 2_usize.pow(self.nb_qubits as u32);

		let vectorization = 1u8; // Haven't bothered yet to try with SIMD cause I don't know how to approach it for now

		// Compute dispatch that respects device limits
		let groups = n; // 1 output row per ABSOLUTE_POS
		let (cube_count, cube_dim) = compute_dispatch(groups);

		let shape_mat = &[n, n, 2usize];
		let shape_vec = &[n, 2usize];

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
				cubecl::frontend::ScalarArg::<u32>::new(n as u32),
			);
		}

		return self.read_vector_out_from_gpu();
	}

	fn run_and_branch(
		&mut self, steps: &[BackendOperation<GPUComplexMatrix>], matrix_on_gpu: Vec<usize>, current_steps: Vec<usize>,
		state: &GPUStateVector, bit_values: &[(usize, u8)], nb_qubits: usize,
	) -> GPUStateVector {
		// If end of program is reached, return the calculated state
		if steps.is_empty() {
			return state.clone();
		}

		match &steps[0] {
			BackendOperation::Matrix(matrix) => {
				let mut new_current_steps = current_steps.clone();
				let last_step = new_current_steps.last_mut().unwrap();
				*last_step += 1;

				let new_state = self.matrix_times_state(matrix, new_current_steps.clone(), state);
				return self.run_and_branch(
					&steps[1..],
					self.matrix_present_on_gpu.clone(),
					current_steps,
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
					let mut state_if_measured = GPUStateVector::zero(nb_qubits);
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
				let mut combined_outcomes = GPUStateVector::zero(nb_qubits);
				// let remaining_operations = &steps[1..];
				for (bit_value, probability, branched_state) in outcomes {
					let mut bit_values: Vec<_> = bit_values.iter().map(|x| *x).collect();
					bit_values.push((*on_qubit, bit_value));

					let mut new_current_steps = current_steps.clone();
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
				let mut new_current_steps = current_steps.clone();
				new_current_steps.push(idx);
				new_steps[0] = BackendOperation::Matrix(gate_to_apply.clone());

				return self.run_and_branch(&new_steps, matrix_on_gpu, new_current_steps, state, bit_values, nb_qubits);
			}
		}
	}
}

impl<R: Runtime> Program for DenseGPUProgram<R> {
	fn run(&mut self, state: &StateVector) -> StateVector {
		let gpu_state = GPUStateVector::from_state(state);
		let operations = self.operations.clone();
		let gpu_final_state = self.run_and_branch(&operations, vec![], vec![0], &gpu_state, &[], self.nb_qubits);
		return gpu_final_state.into_cpu_state();
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	use crate::algorithms::qft_matrix;
	use std::f64::consts::PI;

	#[test]
	// Same test as qft but with matrix multiplication on GPU.
	fn qft_matrix_on_gpu() {
		let frequency = 5;

		for nb_qubits in 5..=12 {
			let dimension = 2_usize.pow(nb_qubits as u32);

			let qft_matrix = qft_matrix(nb_qubits as u32);

			// Prepare state for a wave with given frequency
			let norm = 1.0 / (dimension as f64).sqrt();
			let mut amplitudes = Vec::with_capacity(dimension);
			for x in 0..dimension {
				let angle = -2.0 * PI * (x as f64) / frequency as f64;
				amplitudes.push(Complex::from_polar(norm, angle));
			}
			let state = StateVector::from_vec(amplitudes);

			// Check if total probability is 1
			let total_prob: f64 = (0..dimension).map(|i| state[i].norm_sqr()).sum();
			assert!((total_prob - 1.0).abs() < 1e-12);
			dbg!(&state);

			// Launch matrix-vector multiplication on GPU
			let device = Default::default();
			let final_state = complex_matrix_mul_vector_host::<cubecl::wgpu::WgpuRuntime>(&device, &qft_matrix, &state);

			// Check if the final state gives the closest frequency
			let found_frequency = 2_usize.pow(nb_qubits as u32) as f64 / final_state.most_likely_outcome() as f64;
			dbg!(&final_state);
			dbg!(&found_frequency);
			assert!(found_frequency.round_ties_even() == frequency as f64);
		}
	}
}
