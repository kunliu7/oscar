
from qiskit import Aer
import networkx as nx
from qiskit import Aer
from scipy.optimize import minimize
from .qaoa import get_maxcut_qaoa_circuit

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
    noise_model.add_all_qubit_quantum_error(depolarizing_error(2 * p1Q, 1), 'u3')
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p2Q, 2), 'cx')
    return noise_model


def maxcut_obj(x, G):
    """
    Given a bitstring as a solution, this function returns
    the number of edges shared between the two partitions
    of the graph.
    
    Args:
        x: str
           solution bitstring
           
        G: networkx graph
        
    Returns:
        obj: float
             Objective
    """
    obj = 0
    for i, j in G.edges():
        if x[i] != x[j]:
            obj -= 1
            # obj += 1
            
    return obj


def compute_expectation(counts, G):
    
    """
    Computes expectation value based on measurement results
    
    Args:
        counts: dict
                key as bitstring, val as count
           
        G: networkx graph
        
    Returns:
        avg: float
             expectation value
    """
    
    avg = 0
    sum_count = 0
    for bitstring, count in counts.items():
        
        obj = maxcut_obj(bitstring, G)
        avg += obj * count
        sum_count += count
        
    return avg/sum_count


# Finally we write a function that executes the circuit on the chosen backend
def get_expectation(G, noise_model, num_shots):
    
    """
    Runs parametrized circuit
    
    Args:
        G: networkx graph
        p: int,
           Number of repetitions of unitaries
    """
    
    backend = Aer.get_backend('aer_simulator')
    # noise_model = get_depolarizing_error_noise_model(0.001, 0.02)
    # backend.shots = num_shots
    
    def execute_circ(beta_gamma):
        beta = beta_gamma[:len(beta_gamma) // 2]
        gamma = beta_gamma[len(beta_gamma) // 2:]

        # return noisy_qaoa_maxcut_energy(G, beta, gamma, noise_model=None)#noise_model)
        qc = get_maxcut_qaoa_circuit(G, beta, gamma)
        qc.measure_all()
        counts = backend.run(
            qc,
            num_shots=num_shots,
            noise_model=noise_model
        ).result().get_counts()
        
        return -compute_expectation(counts, G)
    
    return execute_circ


def optimize_under_noise(G: nx.Graph, init_beta_gamma, noise_model, num_shots, opt_method):
    expectation = get_expectation(G, noise_model, num_shots)
    params_path = []

    def cb(xk):
        params_path.append(xk)

    rst = minimize(expectation, 
                init_beta_gamma, 
                # method='COBYLA')
                method=opt_method,
                callback=cb)
    print(rst)
    eigenstate = rst.x
    eigenvalue = rst.fun


    return eigenstate, eigenvalue, params_path
