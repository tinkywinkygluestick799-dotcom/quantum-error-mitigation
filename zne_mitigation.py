"""
Zero-Noise Extrapolation (ZNE) for Quantum Error Mitigation
Author: Alishba Zafar
"""

import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
import matplotlib.pyplot as plt

def create_test_circuit(depth):
    """Create a random quantum circuit with given depth"""
    circuit = QuantumCircuit(2, 2)
    for _ in range(depth):
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.rx(np.pi/4, 0)
        circuit.ry(np.pi/4, 1)
    circuit.measure([0,1], [0,1])
    return circuit

def simulate_with_noise(circuit, noise_rate, shots=10000):
    """Run circuit on noisy simulator"""
    noise_model = NoiseModel()
    error = depolarizing_error(noise_rate, 1)
    noise_model.add_all_qubit_quantum_error(error, ['id', 'x', 'y', 'z', 'h', 'rx', 'ry', 'cx'])
    
    backend = Aer.get_backend('qasm_simulator')
    job = execute(circuit, backend, noise_model=noise_model, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    # Calculate error rate (probability of not getting '00')
    correct = counts.get('00', 0)
    error_rate = 1 - (correct / shots)
    return error_rate

def zero_noise_extrapolation(circuit, scale_factors, noise_rate, shots=10000):
    """Apply ZNE by scaling noise and extrapolating to zero"""
    error_rates = []
    
    for factor in scale_factors:
        # Scale the noise
        scaled_noise = noise_rate * factor
        if scaled_noise > 0.5:  # Cap at 0.5
            scaled_noise = 0.5
        
        # Run simulation with scaled noise
        err = simulate_with_noise(circuit, scaled_noise, shots)
        error_rates.append(err)
        print(f"  Scale factor {factor}: Error rate = {err:.4f}")
    
    # Linear extrapolation to zero noise
    coeffs = np.polyfit(scale_factors, error_rates, 1)
    mitigated_error = coeffs[1]  # Intercept at scale factor 0
    
    return mitigated_error, error_rates

# Run experiment
print("=" * 50)
print("Zero-Noise Extrapolation Experiment")
print("=" * 50)

depths = [5, 10, 15, 20, 25]
noise_rate = 0.01
scale_factors = [1, 1.5, 2, 2.5, 3]

unmitigated_errors = []
mitigated_errors = []

for depth in depths:
    print(f"\n--- Circuit Depth: {depth} ---")
    circuit = create_test_circuit(depth)
    
    # Unmitigated (scale factor = 1)
    unmitigated = simulate_with_noise(circuit, noise_rate)
    unmitigated_errors.append(unmitigated)
    print(f"Unmitigated error: {unmitigated:.4f}")
    
    # ZNE mitigation
    mitigated, error_rates = zero_noise_extrapolation(circuit, scale_factors, noise_rate)
    mitigated_errors.append(mitigated)
    print(f"Mitigated error: {mitigated:.4f}")
    print(f"Improvement: {(unmitigated - mitigated) / unmitigated * 100:.1f}%")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(depths, [e*100 for e in unmitigated_errors], 'o-', label='Unmitigated', linewidth=2, markersize=8, color='#dc2626')
plt.plot(depths, [e*100 for e in mitigated_errors], 's-', label='ZNE Mitigated', linewidth=2, markersize=8, color='#10b981')
plt.xlabel('Circuit Depth', fontsize=12)
plt.ylabel('Error Rate (%)', fontsize=12)
plt.title('Zero-Noise Extrapolation Performance', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('zne_results.png', dpi=150)
plt.show()

print("\n" + "=" * 50)
print("Results saved to: zne_results.png")
print("=" * 50)
