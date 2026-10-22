# WLCBC & LCBC — core algorithms

Four original core Python files, selected from the supplied `Core Algorithm Code in LCBC` directory, with a README and dependency list. Archived variants, separate experiment scripts, plots, and system files are excluded. Algorithm text, original filenames, comments, and embedded examples are preserved. GitHub's web editor normalized the Python sources' CRLF line endings to LF; the downloadable local source package retains the original line endings.

| Source | Implementation | Main entry points |
| --- | --- | --- |
| `OTA_LCBC_code_real_new.py` | Real-valued over-the-air LCBC weighted sum-MSE solver. Alternates LMMSE decoder updates and a power-constrained precoder update; includes SVD side-information reduction, per-user noise scaling, and a safeguarded Newton multiplier search. | `User`, `SolverConfig`, `solve_wsmse_ao` |
| `Hua_Jafar_2-user_LCBC_real_new.py` | Two-user real-field LCBC subspace construction and linear encoding/decoding; includes analog AWGN simulation and an analytic MSE predictor. | `design_lcbc_real`, `RealLCBCScheme`, `simulate_real_lcbc`, `predict_mse_real_lcbc` |
| `Yu_Jafar_3-user_LCBC_real.py` | Three-user real-field LCBC construction: redundancy removal, subspace decomposition, rational LP vertex enumeration, matrix extension, and random mixing. Uses fixed linear decoders in AWGN simulation. | `build_lcbc_3user_code_real`, `simulate_lcbc_real` |
| `3_user_finite_LCBC_code.py` | Three-user prime-field LCBC construction with extension-field multiplication matrices; includes p-PAM modulation, AWGN, hard decisions, and symbol-error simulation. | `build_lcbc_3user_code`, `simulate_lcbc` |

## Construction attribution

The two-user source credits *The Capacity of Computation Broadcast* by Hua Sun and Syed A. Jafar. The three-user real-field source header credits “Yao & Jafar”; that source attribution is retained without expanding the name. These companion LCBC constructions are distinct from the primary WLCBC weighted sum-MSE optimizer. Original source comments and author credits remain unchanged.

## Dependencies and use

The sources import NumPy and PyTorch. Matplotlib is optional for the OTA solver's plotting function. `requirements.txt` lists these external dependencies without version pins: no tested environment or lockfile was supplied. Standard-library modules need no separate installation.

Use a project-local environment; do not install research dependencies into the system Python. All four files place their embedded demos behind `if __name__ == "__main__"`; direct execution runs those demos. Import the files to call the core entry points with your own matrices. Original filenames containing hyphens or an initial digit can be loaded with `importlib.util.spec_from_file_location`.

The OTA solver's command-line interface accepts `--device cpu` and `--no-plot`; without `--no-plot`, it saves a curve image (`mse_curve.png` by default). Other embedded demos print numerical summaries. The OTA preprocessor updates the supplied `User` objects; its initializer can set the global PyTorch seed.

## Scope and verification

**Certified source integrity:** the four local source copies match their supplied originals byte for byte, and each file was parsed for Python syntax. GitHub blob hashes confirm that the published Python files match those originals after CRLF-to-LF line-ending normalization. No algorithm text was changed. The public repository contains only the four Python sources, this README, and `requirements.txt`.

**Source inspection only:** dependencies were not installed, algorithms and demos were not executed, and no numerical or performance results were computed. This package does not establish theorem correctness, global optimality of the alternating solver, real-time operation, hardware deployment, or benchmark performance. The real-field constructions depend on numerical rank tolerances; the three-user constructions use bounded randomized search and may fail for a given input or configuration. Original comments that refer to paper theorems or optimality are preserved as supplied, without independent verification.

These are algorithm implementations and simulation building blocks for research prototypes. No RF hardware interface is included. No license was supplied in the selected source set, so this package does not add or infer a license.
