use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use qualcul::Gate;
use qualcul::algorithms::qft_matrix;
use qualcul::backend::dense_cpu::{DenseCPUBackend, DenseCPUProgram};
use qualcul::backend::{Backend, Program};
use qualcul::circuit::Circuit;
use qualcul::{circuit::StateVector, state::Ket};
use std::hint::black_box;
use std::time::Duration;

fn qft_bench(c: &mut Criterion) {
	let mut group = c.benchmark_group("qft");
	group.sample_size(10);
	group.warm_up_time(Duration::from_secs(1));
	group.measurement_time(Duration::from_secs(5));

	let backends = ["backend_1", "backend_2"];

	for backend in backends.iter() {
		for nb_qubits in 2..=6 {
			let dimension = 2usize.pow(nb_qubits as u32);
			let state = StateVector::from_ket(&Ket::base(0, dimension));

			let circuit = DenseCPUProgram::from_matrix(qft_matrix(nb_qubits));

			let id = BenchmarkId::new(*backend, nb_qubits);
			group.bench_with_input(id, &(backend, nb_qubits), |b, _| {
				let circuit = match *backend {
					"backend_1" => &circuit,
					"backend_2" => &circuit,
					_ => panic!("Unknown backend"),
				};

				b.iter(|| {
					circuit.run(black_box(&state));
				});
			});
		}
	}
	group.finish();
}

fn ghz_n_bench(c: &mut Criterion) {
	let mut group = c.benchmark_group("ghz_n");
	group.sample_size(10);
	group.warm_up_time(Duration::from_secs(1));
	group.measurement_time(Duration::from_secs(5));

	let backends = ["backend_1", "backend_2"];

	for backend in backends.iter() {
		for nb_qubits in 2..=8 {
			let nb_dimensions = 2 << nb_qubits; // 2^(n+1)
			let state = StateVector::from_ket(&Ket::random(nb_dimensions));

			let mut circuit = Circuit::new(nb_qubits + 1).then(Gate::h().on(0));
			for i in 0..nb_qubits {
				circuit = circuit.then(Gate::x().on(i + 1).control(vec![i]));
			}
			let circuit = DenseCPUBackend.compile(&circuit);

			let id = BenchmarkId::new(*backend, nb_qubits);
			group.bench_with_input(id, &(backend, nb_qubits), |b, _| {
				let circuit = match *backend {
					"backend_1" => &circuit,
					"backend_2" => &circuit,
					_ => panic!("Unknown backend"),
				};

				b.iter(|| {
					circuit.run(black_box(&state));
				});
			});
		}
	}
	group.finish();
}

criterion_group!(benches, qft_bench, ghz_n_bench);
criterion_main!(benches);
