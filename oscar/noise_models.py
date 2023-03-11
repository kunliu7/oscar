

from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors.standard_errors import depolarizing_error, pauli_error


def get_pauli_error_noise_model(p_error: float):
    noise_model = NoiseModel()

    bit_flip = pauli_error([('X', p_error), ('I', 1 - p_error)])
    phase_flip = pauli_error([('Z', p_error), ('I', 1 - p_error)])
    print(bit_flip)
    print(phase_flip)
    bitphase_flip = bit_flip.compose(phase_flip)
    print(bitphase_flip)
    noise_model.add_all_qubit_quantum_error(bitphase_flip, ['u1', 'u2', 'u3'])
    # noise_model.add_all_qubit_quantum_error(bitphase_flip, ['cx'])
    return noise_model


def get_depolarizing_error_noise_model(p1Q: float, p2Q: float):
    noise_model = NoiseModel()

    # Depolarizing error on the gates u2, u3 and cx (assuming the u1 is virtual-Z gate and no error)
    # p1Q = 0.002
    # p2Q = 0.01

    noise_model.add_all_qubit_quantum_error(depolarizing_error(p1Q, 1), 'u2')
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(2 * p1Q, 1), 'u3')
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p2Q, 2), 'cx')
    return noise_model
