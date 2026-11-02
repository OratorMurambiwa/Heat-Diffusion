#include "comparison_common.hpp"
#include "comparison_build.hpp"
#include "cuda_timing.hpp"
#include "solver.hpp"
#include <omp.h>
#include <unistd.h>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>

namespace {
int positive(const char* text) {
    std::string value(text);
    size_t used = 0;
    int number = std::stoi(value, &used);
    if (used != value.size() || number <= 0) throw std::runtime_error("Expected positive integer: " + value);
    return number;
}
std::string env(const char* name) {
    const char* value = std::getenv(name);
    return value ? value : "unset";
}
std::string cpu_model() {
    std::ifstream source("/proc/cpuinfo");
    std::string line;
    while (std::getline(source, line))
        if (line.rfind("model name", 0) == 0) return line.substr(line.find(':') + 2);
    return "unknown";
}
std::string number(double value) {
    std::ostringstream stream;
    stream << std::setprecision(17) << value;
    return stream.str();
}
void csv_row(std::ofstream& file, const std::vector<std::string>& fields) {
    for (size_t i = 0; i < fields.size(); ++i) {
        if (i) file << ',';
        file << Comparison::csv_string(fields[i]);
    }
    file << '\n';
    file.flush();
    if (!file) throw std::runtime_error("CSV write failed");
}
}

int main(int argc, char** argv) {
    try {
        int iterations = 100, warmup = 10, repeats = 9, threads = 8;
        bool cpu_only = false;
        std::vector<GpuInitialization> initialization_modes{GpuInitialization::TwoHostCopies};
        std::vector<GpuKernel> kernels{GpuKernel::Naive};
        std::vector<int> sizes{64, 128, 256};
        std::string path = "cpu_cuda_benchmark.csv";
        for (int i = 1; i < argc; ++i) {
            const std::string option(argv[i]);
            if (option == "--help") {
                std::cout << "Usage: cpu_cuda_benchmark [--threads N --iterations N --warmup N --repeats N]\n"
                          << " [--csv PATH] [--size N] [--cpu-only] [--gpu-init baseline|d2d|both]\n"
                          << " [--gpu-kernel naive|z-register|both]\n"
                          << "Default sweep: 64^3, 128^3, 256^3; FP64; 100 iterations, 10 warmups, 9 repeats.\n"
                          << "--size overrides the sweep. CSV is overwritten. --cpu-only produces no GPU results.\n"
                          << "Default: naive kernel, baseline (two H2D copies); multiple variants alternate order each repeat.\n";
                return 0;
            }
            if (option == "--cpu-only") { cpu_only = true; continue; }
            if (i + 1 == argc) throw std::runtime_error("Missing value for " + option);
            const char* value = argv[++i];
            if (option == "--threads") threads = positive(value);
            else if (option == "--iterations") iterations = positive(value);
            else if (option == "--warmup") warmup = positive(value);
            else if (option == "--repeats") repeats = positive(value);
            else if (option == "--size") sizes = {positive(value)};
            else if (option == "--csv") path = value;
            else if (option == "--gpu-init") {
                const std::string mode(value);
                if (mode == "baseline") initialization_modes = {GpuInitialization::TwoHostCopies};
                else if (mode == "d2d") initialization_modes = {GpuInitialization::HostThenDeviceCopy};
                else if (mode == "both") initialization_modes = {
                    GpuInitialization::TwoHostCopies, GpuInitialization::HostThenDeviceCopy};
                else throw std::runtime_error("--gpu-init must be baseline, d2d, or both");
            }
            else if (option == "--gpu-kernel") {
                const std::string kernel(value);
                if (kernel == "naive") kernels = {GpuKernel::Naive};
                else if (kernel == "z-register") kernels = {GpuKernel::ZRegister};
                else if (kernel == "both") kernels = {GpuKernel::Naive, GpuKernel::ZRegister};
                else throw std::runtime_error("--gpu-kernel must be naive, z-register, or both");
            }
            else throw std::runtime_error("Unknown option: " + option);
        }
        if (repeats < 2) throw std::runtime_error("Need at least two repeats");
        for (int n : sizes) {
            if (n < 3 || n > std::numeric_limits<int>::max() / n ||
                static_cast<size_t>(n) * n > std::vector<double>().max_size() / n)
                throw std::runtime_error("Invalid grid size for Grid3D");
        }
#ifndef HEAT_BENCH_HAS_CUDA
        if (!cpu_only) throw std::runtime_error("Built without CUDA: use --cpu-only or configure ENABLE_CUDA=ON");
#endif
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
        int actual_threads = 0;
        #pragma omp parallel
        {
            #pragma omp single
            actual_threads = omp_get_num_threads();
        }
        if (actual_threads != threads) throw std::runtime_error("OpenMP team smaller than requested");
        char hostname[256]{};
        if (gethostname(hostname, sizeof(hostname) - 1) != 0) throw std::runtime_error("gethostname failed");
        const std::string cpu_name = cpu_model();
        std::ofstream csv(path);
        if (!csv) throw std::runtime_error("Cannot open CSV: " + path);
        csv_row(csv, {"nx","ny","nz","precision","threads","iterations","warmup","repeats",
            "timing_type","timer","median_seconds","min_seconds","max_seconds","mean_seconds",
            "stddev_seconds","cell_updates","updates_per_second","speedup_vs_cpu_median",
            "max_error_K","validation_reference","samples_seconds","hostname","slurm_job_id",
            "cpu_model","gpu_model","gpu_pci_bus_id","gpu_compute_capability","gpu_memory_bytes",
            "cuda_runtime_version","cuda_driver_version","cxx_compiler","cxx_flags","cuda_compiler",
            "cuda_flags","cuda_host_compiler","cuda_architectures","build_type","omp_places","omp_proc_bind",
            "initialization_method","variant_order","speedup_vs_two_h2d_median",
            "timed_h2d_bytes","timed_d2d_bytes","timed_d2h_bytes",
            "kernel_variant","z_cells_per_thread","speedup_vs_naive_median","time_reduction_vs_naive_percent"});
        using Clock = std::chrono::steady_clock;
        for (int n : sizes) {
            std::cout << "Starting " << n << "^3, " << threads << " CPU threads" << std::endl;
            Grid3D initial(n, n, n, Config::AMBIENT_TEMP);
            Comparison::initialize(initial);
            Grid3D current = initial, next = initial;
            std::vector<double> cpu_samples;
            double cpu_error = 0.0;
            auto advance_cpu = [&](int steps) {
                for (int step = 0; step < steps; ++step) {
                    Solver::compute_diffusion(current, next);
                    current.data.swap(next.data);
                }
            };
            for (int repeat = 0; repeat < repeats; ++repeat) {
                current.data = initial.data; next.data = initial.data;
                advance_cpu(warmup);
                // Warmups heat the execution path, but do not advance timed initial data.
                current.data = initial.data; next.data = initial.data;
                const auto start = Clock::now();
                advance_cpu(iterations);
                const auto stop = Clock::now();
                cpu_samples.push_back(std::chrono::duration<double>(stop - start).count());
                cpu_error = std::max(cpu_error, Comparison::check_cpu(current, iterations));
            }
            std::vector<GpuMeasurements> gpu_results;
#ifdef HEAT_BENCH_HAS_CUDA
            if (!cpu_only) gpu_results = measure_gpu(initial, current, iterations, warmup, repeats, initialization_modes, kernels);
#endif
            const GpuMeasurements empty_gpu;
            const auto& gpu = gpu_results.empty() ? empty_gpu : gpu_results.front();
            const Comparison::Statistics cpu_statistics(cpu_samples);
            const double updates = static_cast<double>(n - 2) * (n - 2) * (n - 2) * iterations;
            auto save = [&](const std::string& type, const std::string& timer,
                            const std::vector<double>& samples, double error, const std::string& reference,
                            const GpuMeasurements* measurement) {
                const Comparison::Statistics statistics(samples);
                const bool transfer_inclusive = type == "gpu_transfer_inclusive";
                const bool baseline = measurement && measurement->initialization == GpuInitialization::TwoHostCopies;
                double baseline_time = 0.0, naive_time = 0.0;
                if (measurement) {
                    for (const auto& candidate : gpu_results) {
                        const bool init_reference = candidate.kernel == measurement->kernel &&
                            candidate.initialization == GpuInitialization::TwoHostCopies;
                        const bool kernel_reference = candidate.initialization == measurement->initialization &&
                            candidate.kernel == GpuKernel::Naive;
                        if (init_reference || kernel_reference) {
                            const double median = Comparison::Statistics(transfer_inclusive ?
                                candidate.end_to_end_seconds : candidate.kernel_seconds).median;
                            if (init_reference) baseline_time = median;
                            if (kernel_reference) naive_time = median;
                        }
                    }
                }
                const double field_bytes = static_cast<double>(initial.data.size()) * sizeof(double);
                std::vector<std::string> fields{
                    std::to_string(n),std::to_string(n),std::to_string(n),"FP64",std::to_string(actual_threads),
                    std::to_string(iterations),std::to_string(warmup),std::to_string(repeats),type,timer,
                    number(statistics.median),number(statistics.minimum),number(statistics.maximum),
                    number(statistics.mean),number(statistics.stddev),number(updates),number(updates/statistics.median),
                    number(cpu_statistics.median/statistics.median),number(error),reference,
                    Comparison::samples_string(samples),hostname,env("SLURM_JOB_ID"),cpu_name,gpu.device,
                    gpu.pci_bus_id,cpu_only ? "" : std::to_string(gpu.compute_major)+"."+std::to_string(gpu.compute_minor),
                    cpu_only ? "" : std::to_string(gpu.global_memory_bytes),
                    cpu_only ? "" : std::to_string(gpu.runtime_version),cpu_only ? "" : std::to_string(gpu.driver_version),
                    comparison_cxx,comparison_cxx_flags,comparison_cuda,comparison_cuda_flags,
                    comparison_cuda_host,comparison_cuda_arch,comparison_build_type,env("OMP_PLACES"),env("OMP_PROC_BIND"),
                    measurement ? initialization_name(measurement->initialization) : "not_applicable",
                    cpu_only ? "cpu_only" : kernels.size() > 1 ?
                        "alternating_reverse_init_then_kernel_cartesian_order" : initialization_modes.size() == 2 ?
                        "alternating_two_h2d_first_even_repeats" : "single_variant",
                    measurement && baseline_time > 0.0 ? number(baseline_time/statistics.median) : "",
                    number(transfer_inclusive ? (baseline ? 2.0 : 1.0)*field_bytes : 0.0),
                    number(transfer_inclusive && !baseline ? field_bytes : 0.0),
                    number(transfer_inclusive ? field_bytes : 0.0),
                    measurement ? kernel_name(measurement->kernel) : "not_applicable",
                    measurement ? std::to_string(measurement->kernel == GpuKernel::ZRegister ? n - 2 : 1) : "",
                    naive_time > 0.0 ? number(naive_time/statistics.median) : "",
                    naive_time > 0.0 ? number(100.0*(1.0-statistics.median/naive_time)) : ""};
                csv_row(csv, fields);
                std::cout << std::setprecision(8) << type
                          << " [" << (measurement ? initialization_name(measurement->initialization) : "CPU")
                          << (measurement ? std::string("/")+kernel_name(measurement->kernel) : "")
                          << "]: median=" << statistics.median
                          << " s, stddev=" << statistics.stddev << " s, updates/s=" << updates/statistics.median
                          << ", speedup=" << cpu_statistics.median/statistics.median
                          << ", validation max error=" << error << " K\n";
            };
            save("cpu_batch", "steady_clock", cpu_samples, cpu_error, "discrete_sine", nullptr);
            for (const auto& measurement : gpu_results) {
                save("gpu_kernel_batch", "cuda_events", measurement.kernel_seconds, measurement.kernel_error, "cpu", &measurement);
                save("gpu_transfer_inclusive", "steady_clock", measurement.end_to_end_seconds, measurement.end_to_end_error, "cpu", &measurement);
            }
        }
        csv.close();
        if (!csv) throw std::runtime_error("CSV close failed");
        std::cout << "PASS: all measured fields validated. CSV: " << path << '\n';
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
