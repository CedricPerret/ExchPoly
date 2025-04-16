# 💱 ExchPoly

Code and derivations for the study:

**"How economic exchange can favour human genetic diversity"**

This repository includes both simulation scripts and analytical work supporting the study.

---

## 📖 Overview

This repository contains two main components:

### 1. 🧬 Agent-based Simulations (`Simulations_ExchPoly.jl`)
Simulation code written in Julia, using `Toolbox.jl` functions (from [`JuliassicPark.jl`](https://github.com/CedricPerret/JuliassicPark)). These simulations explore how modes of economic exchange affect the evolution of individual traits under different parameter regimes.

Scenarios vary:
- Modes of exchange
- Values of `σ` and `α`
- Parameter sweeps over `η` and `σ`

The structure of the code mirrors standard academic modelling: it starts by defining the fitness function — which links traits to evolutionary success — followed by components required to run the simulations.

> **Note:** All relevant functions must be defined and loaded at the start of the script to ensure the simulations run correctly.

### 2. 📐 Analytical Derivations (`ExchPoly_analysis_mathematica.nb`)
A Mathematica notebook containing all derivations and analytical results referenced in the paper.

---

## 📦 Dependencies

**For the Julia simulations:**
- Julia 1.9+
- External repo: [`JuliassicPark.jl`](https://github.com/CedricPerret/JuliassicPark)
- Common Julia packages: `Optim.jl`, `Plots.jl`, etc.

**For the analytical notebook:**
- Wolfram Mathematica 12.0+

---

## 🚧 Status

This codebase is currently **not tailored for public use** and may require manual setup or editing to run on your system.  
If you're interested in reproducing or adapting the simulations or analysis, feel free to **contact me with any questions**.

---

## 🧠 Citation

If you use this code or adapt the model, please cite the associated paper once published.

---

## License

Currently private use only. Reach out if you’re interested in using or extending it.
