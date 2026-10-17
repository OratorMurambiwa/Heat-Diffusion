
from pathlib import Path
from math import sqrt
from compare_mpi import read_vtk

fp64 = Path("mpi_validation_59886188/parallel")
fp32 = Path("mpi_validation_59886595/parallel")

print("Step       Max Error (K)       RMS Error (K)")

for step in range(0, 1000, 100):
    max_error = 0.0
    squared_error = 0.0
    count = 0

    for rank in range(4):
        filename = f"output_step_{step}_rank_{rank}.vtk"

        ref = read_vtk(fp64 / filename)
        test = read_vtk(fp32 / filename)

        if ref[:3] != test[:3]:
            raise ValueError("Grid mismatch")

        for a, b in zip(ref[3], test[3]):
            error = abs(a - b)
            max_error = max(max_error, error)
            squared_error += error * error
            count += 1

    rms_error = sqrt(squared_error / count)

    print(
        f"{step:<10} "
        f"{max_error:.8e}       "
        f"{rms_error:.8e}"
    )


