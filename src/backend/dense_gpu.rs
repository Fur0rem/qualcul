// Implementation notes:
// Since I couldn't get CubeCL to work with a custom Complex type, I decided to transform the complex vectors/matrices into a tensor with 1 more dimension of size 2,
// which represent the real/imaginary part.
// For example:
// - matrix: shape [n, n] -> tensor of shape [n, n, 2] (re, im)
// - vector: shape [n] -> tensor of shape [n, 2] (re, im)

use crate::{ComplexMatrix, circuit::StateVector};

use cubecl::prelude::*;
use num::{Complex, complex::ComplexFloat};

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

fn matrix_into_split_f64(mat: &ComplexMatrix) -> Vec<f64> {
	let n = mat.size_side();
	let mut out = Vec::with_capacity(n * n * 2);
	for r in 0..n {
		for c in 0..n {
			let v = mat[(r, c)];
			out.push(v.re);
			out.push(v.im);
		}
	}
	return out;
}

fn state_into_split_f64(vec: &StateVector) -> Vec<f64> {
	let n = vec.components().len();
	let mut out = Vec::with_capacity(n * 2);
	for i in 0..n {
		let v = vec.components()[i];
		out.push(v.re());
		out.push(v.im());
	}
	return out;
}

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
	let mat_f64 = matrix_into_split_f64(matrix);
	let vec_f64 = state_into_split_f64(vec_in);

	let mat_bytes = bytes_from_f64_slice(&mat_f64);
	let vec_bytes = bytes_from_f64_slice(&vec_f64);

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
