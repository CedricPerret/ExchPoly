# 💱 ExchPoly

Code for agent-based simulations presented in:

**"How economic exchange can favour human genetic diversity"**

This repository contains the simulation scripts and analysis used in the above study.

## 📖 Overview

This repository starts by defining the fitness function, which is then passed to the `evol_model` function from `Toolbox.jl` (in [`JuliassicPark.jl`](https://github.com/CedricPerret/JuliassicPark.jl)) along with simulation parameters. The code includes a variety of simulation scenarios, varying:
- Modes of exchange
- Values of `σ` and `α`
- Parameter sweeps over `η` and `σ`

The structure of the code mirrors how models are typically introduced in academic papers. It starts by defining the fitness function — which links individual traits to evolutionary success — followed by the components required to run the simulations.

**Note:** All relevant functions must be defined and loaded at the start of the script to ensure the simulations run correctly.

---

## 📦 Dependencies

This project **requires functions defined in [`JuliassicPark.jl`](https://github.com/CedricPerret/JuliassicPark.jl)**. Make sure to clone and load that repository as part of your environment before running the simulations.

---

## 🚧 Status

The code is **not currently tailored for public use** and may require manual setup or editing to run on your system.  
If you're interested in running or adapting the code, feel free to **contact me with any questions**.

---

## 🔧 Requirements

- Julia 1.9+
- External repo: [`JuliassicPark.jl`](https://github.com/yourusername/JuliassicPark.jl)
- Common Julia packages: `Optim.jl`, `Plots.jl`, etc.

---

## 🧠 Citation

If you use this code or adapt the model, please cite the associated paper once published.

---

## License

Currently private use only. Reach out if you’re interested in using or extending it.

