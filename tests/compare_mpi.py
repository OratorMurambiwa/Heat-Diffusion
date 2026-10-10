from array import array
from pathlib import Path
import math
import sys


def read_vtk(path):
    dimensions = origin = spacing = point_count = None

    with path.open() as file:
        for line in file:
            parts = line.split()
            if not parts:
                continue

            if parts[0] == "DIMENSIONS":
                dimensions = tuple(map(int, parts[1:]))
            elif parts[0] == "ORIGIN":
                origin = tuple(map(float, parts[1:]))
            elif parts[0] == "SPACING":
                spacing = tuple(map(float, parts[1:]))
            elif parts[0] == "POINT_DATA":
                point_count = int(parts[1])
            elif parts[:2] == ["LOOKUP_TABLE", "default"]:
                break
        else:
            raise ValueError(f"Missing temperature data: {path}")

        values = array("d", (
            float(line) for line in file if line.strip()
        ))

    if any(value is None for value in
           (dimensions, origin, spacing, point_count)):
        raise ValueError(f"Incomplete VTK header: {path}")

    expected_count = math.prod(dimensions)
    if len(values) != expected_count or point_count != expected_count:
        raise ValueError(f"Incorrect data count: {path}")

    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Nonfinite temperature: {path}")

    return dimensions, origin, spacing, values


def compare(folder):
    serial_path = folder / "serial/output_step_900_rank_0.vtk"
    dims, origin, spacing, reference = read_vtk(serial_path)

    parallel_paths = list(
        (folder / "parallel").glob("output_step_900_rank_*.vtk")
    )
    if len(parallel_paths) != 4:
        raise ValueError(
            f"Expected four parallel files, found {len(parallel_paths)}"
        )

    plane_size = dims[0] * dims[1]
    covered_layers = set()
    max_error = 0.0

    for path in sorted(parallel_paths):
        local_dims, local_origin, local_spacing, values = read_vtk(path)

        if local_dims[:2] != dims[:2]:
            raise ValueError(f"X/Y dimensions differ: {path}")

        if local_spacing != spacing:
            raise ValueError(f"Grid spacing differs: {path}")

        if local_origin[:2] != origin[:2]:
            raise ValueError(f"X/Y origin differs: {path}")

        layer_position = (local_origin[2] - origin[2]) / spacing[2]
        start_layer = round(layer_position)

        if not math.isclose(
            layer_position, start_layer, rel_tol=0.0, abs_tol=1e-8
        ):
            raise ValueError(f"Z origin is not aligned: {path}")

        layers = set(range(start_layer, start_layer + local_dims[2]))

        if start_layer < 0 or start_layer + local_dims[2] > dims[2]:
            raise ValueError(f"Slab extends outside the global grid: {path}")

        if covered_layers & layers:
            raise ValueError(f"Overlapping real layers: {path}")

        covered_layers.update(layers)
        start_index = start_layer * plane_size

        local_error = max(
            abs(value - reference[start_index + index])
            for index, value in enumerate(values)
        )

        max_error = max(max_error, local_error)
        print(f"{path.name}: max difference = {local_error:.6e} K")

    if covered_layers != set(range(dims[2])):
        raise ValueError("Parallel output does not cover the entire grid")

    tolerance = 1e-8
    print(f"\nOverall maximum difference: {max_error:.6e} K")

    if max_error > tolerance:
        raise ValueError(
            f"MPI results differ by more than {tolerance} K"
        )

    print("PASS: one-rank and four-rank temperature fields match.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python3 tests/compare_mpi.py mpi_validation_JOB_ID"
        )

    try:
        compare(Path(sys.argv[1]))
    except (OSError, ValueError) as error:
        raise SystemExit(f"FAIL: {error}")
