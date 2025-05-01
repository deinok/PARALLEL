import numpy as np
import matplotlib.pyplot as plt

# Data from the tables

# BT Benchmark
BT_serial = {"A": 48.01, "B": 204.14, "C": 847.66}
BT_OpenMP = {"A": [25.64, 15.19, 15.91], "B": [109.97, 66.93, 69.20], "C": [469.10, 282.62, 285.26]}
BT_MPI = {"A": [55.34, 18.95, 38.06, 50.24], "B": [230.09, 75.80, 89.00, 75.93], "C": [955.23, 320.41, 206.12, 146.26]}

OpenMP_threads = [2, 4, 8]
MPI_processes = [1, 4, 9, 16]

# IS Benchmark
IS_serial = {"A": 0.41, "B": 1.83, "C": 8.89}
IS_OpenMP = {"A": [0.26, 0.15, 0.16], "B": [1.15, 0.63, 0.63], "C": [5.96, 3.12, 3.13]}
IS_MPI = {"A": [0.27, 0.20, 0.82, 4.88, 6.54], "B": [1.06, 0.79, 3.27, 6.12, 9.39], "C": [5.57, 3.25, 12.98, 13.07, 9.70]}

MPI_processes_IS = [2, 4, 8, 16, 32]

# Function to calculate SpeedUp and Efficiency
def calculate_speedup_efficiency(serial_time, parallel_times, num_threads):
    speedup = [serial_time / t for t in parallel_times]
    efficiency = [s / n for s, n in zip(speedup, num_threads)]
    return speedup, efficiency

# Compute SpeedUp and Efficiency for BT
BT_speedup_OpenMP, BT_efficiency_OpenMP = {}, {}
BT_speedup_MPI, BT_efficiency_MPI = {}, {}

for cls in BT_serial.keys():
    BT_speedup_OpenMP[cls], BT_efficiency_OpenMP[cls] = calculate_speedup_efficiency(BT_serial[cls], BT_OpenMP[cls], OpenMP_threads)
    BT_speedup_MPI[cls], BT_efficiency_MPI[cls] = calculate_speedup_efficiency(BT_serial[cls], BT_MPI[cls], MPI_processes)

# Compute SpeedUp and Efficiency for IS
IS_speedup_OpenMP, IS_efficiency_OpenMP = {}, {}
IS_speedup_MPI, IS_efficiency_MPI = {}, {}

for cls in IS_serial.keys():
    IS_speedup_OpenMP[cls], IS_efficiency_OpenMP[cls] = calculate_speedup_efficiency(IS_serial[cls], IS_OpenMP[cls], OpenMP_threads)
    IS_speedup_MPI[cls], IS_efficiency_MPI[cls] = calculate_speedup_efficiency(IS_serial[cls], IS_MPI[cls], MPI_processes_IS)

# Plot SpeedUp and Efficiency
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("SpeedUp and Efficiency for BT and IS Benchmarks")

# Plot BT SpeedUp
for cls in BT_serial.keys():
    axes[0, 0].plot(OpenMP_threads, BT_speedup_OpenMP[cls], marker='o', linestyle='-', label=f"BT-{cls} (OpenMP)")
    axes[0, 0].plot(MPI_processes, BT_speedup_MPI[cls], marker='s', linestyle='--', label=f"BT-{cls} (MPI)")

axes[0, 0].set_title("BT SpeedUp")
axes[0, 0].set_xlabel("Threads/Processes")
axes[0, 0].set_ylabel("SpeedUp")
axes[0, 0].legend()
axes[0, 0].grid()

# Plot IS SpeedUp
for cls in IS_serial.keys():
    axes[0, 1].plot(OpenMP_threads, IS_speedup_OpenMP[cls], marker='o', linestyle='-', label=f"IS-{cls} (OpenMP)")
    axes[0, 1].plot(MPI_processes_IS, IS_speedup_MPI[cls], marker='s', linestyle='--', label=f"IS-{cls} (MPI)")

axes[0, 1].set_title("IS SpeedUp")
axes[0, 1].set_xlabel("Threads/Processes")
axes[0, 1].set_ylabel("SpeedUp")
axes[0, 1].legend()
axes[0, 1].grid()

# Plot BT Efficiency
for cls in BT_serial.keys():
    axes[1, 0].plot(OpenMP_threads, BT_efficiency_OpenMP[cls], marker='o', linestyle='-', label=f"BT-{cls} (OpenMP)")
    axes[1, 0].plot(MPI_processes, BT_efficiency_MPI[cls], marker='s', linestyle='--', label=f"BT-{cls} (MPI)")

axes[1, 0].set_title("BT Efficiency")
axes[1, 0].set_xlabel("Threads/Processes")
axes[1, 0].set_ylabel("Efficiency")
axes[1, 0].legend()
axes[1, 0].grid()

# Plot IS Efficiency
for cls in IS_serial.keys():
    axes[1, 1].plot(OpenMP_threads, IS_efficiency_OpenMP[cls], marker='o', linestyle='-', label=f"IS-{cls} (OpenMP)")
    axes[1, 1].plot(MPI_processes_IS, IS_efficiency_MPI[cls], marker='s', linestyle='--', label=f"IS-{cls} (MPI)")

axes[1, 1].set_title("IS Efficiency")
axes[1, 1].set_xlabel("Threads/Processes")
axes[1, 1].set_ylabel("Efficiency")
axes[1, 1].legend()
axes[1, 1].grid()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
