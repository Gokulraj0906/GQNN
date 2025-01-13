from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
 
# Initialize account.
service = QiskitRuntimeService()
 
# Prepare inputs.
psi = RealAmplitudes(num_qubits=2, reps=2)
H1 = SparsePauliOp.from_list([("II", 1), ("IZ", 2), ("XI", 3)])
theta = [0, 1, 1, 2, 3, 5]
# Bell Circuit
qr = QuantumRegister(2, name="qr")
cr = ClassicalRegister(2, name="cr")
qc = QuantumCircuit(qr, cr, name="bell")
qc.h(qr[0])
qc.cx(qr[0], qr[1])
qc.measure(qr, cr)
 
backend = service.least_busy(operational=True, simulator=False)
pm = generate_preset_pass_manager(target=backend.target, optimization_level=1)
 
bell_isa_circuit = pm.run(qc)
psi_isa_circuit = pm.run(psi)
isa_observables = H1.apply_layout(psi_isa_circuit.layout)
 
with Session(backend=backend) as session:
    # Submit a request to the Sampler primitive within the session.
    sampler = Sampler(mode=session)
    job = sampler.run([bell_isa_circuit])
    pub_result = job.result()[0]
    print(f"Counts: {pub_result.data.cr.get_counts()}")
 
    # Submit a request to the Estimator primitive within the session.
    estimator = Estimator(mode=session)
    estimator.options.resilience_level = 1  # Set options.
    job = estimator.run(
        [(psi_isa_circuit, isa_observables, theta)]
    )
    pub_result = job.result()[0]
    print(f"Expectation values: {pub_result.data.evs}")