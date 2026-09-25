# AGENTS.md

## Coding guidelines

Write code that is fast, simple, and explicit.

### Priorities

1. Optimize for runtime performance when it does not make the code harder to understand.
2. Avoid unnecessary dependencies.
3. Keep control flow easy to follow.

### Style

- Do not use `auto`, use explicit typing.
- Keep existing comment stype.

### Performance

- Avoid unnecessary allocations.
- Try to fuse deep copies when possible and logical.
- When multiple deep copies are for "reseting" a variable, prefer to use a Kokkos parallel_for to set them manually (fewer launches and greater parallelism).
- Avoid repeated work.
- Prefer stack allocation where practical.
- Measure before adding complex optimizations.

### Simplicity

- Use the standard library before adding dependencies.
- Keep APIs small.
- Keep files focused.
- Remove unused code.
- Prefer readable code over overly abstract code.

### Benchmarking

- Run "source ~/toggle_kokkos_profiling.sh" to enable Kokkos profiling
- Only run ONE simulation at a time. When running multiple, run them sequentially.
- Navitate into "Benchmarks" and then navigate to a subdirectory and run "../../build/apps/Main ParamInput.txt"
- Run all Benchmarks available
- Only run a Benchmark for a maximum of 10s. If it doesn't finish in this time, it is because of a bug and the process needs to be killed.
- Save logs of before and after timings in Agents.log in the main folder. Record total time and kernel time for any functions that were changed.

### Review checklist

Before submitting code, verify:

- No `auto` is used.
- Types are explicit.
- The solution uses minimal machinery.
- The code is fast enough for expected workloads.
- No unnecessary abstraction or dependency was added.