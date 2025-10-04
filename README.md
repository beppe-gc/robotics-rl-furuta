# Robotics RL Project – Furuta Pendulum Swing-Up with PPO

This project applies both **Optimal Trajectory Planning** and **Reinforcement Learning (RL)** to control a **Furuta pendulum** (torque-driven inverted pendulum on a rotating base). The task is to swing up the pendulum and balance it in the upright position using both approaches and to compare the results. The project was a partial deliverable for the Robotics (B-KUL-H02A4A) elective at KU Leuven.

## Authors
This project was developed as a group project for the KU Leuven Robotics course (B-KUL-H02A4A) in 2025, in collaboration with another student.

## Project Structure
- `Robotics_RL_FurutaPendulum.ipynb` – main Jupyter notebook with reasoning, experiments and results  
- `src/` – source code
  - `envs/` – custom Gymnasium environments for the Furuta pendulum  
  - `agents/` – RL agents (PPO implementation)  
  - `utils/` – helper scripts for dynamics 
  - `pendulum_description/` – system model and description files  
- `docs/` – original project description  
- `data/` – learned models and training tensorboards  

## Setup
Clone the repo:

```bash
git clone https://github.com/beppe-gc/robotics-rl-furuta.git
cd robotics-rl-furuta
```

## Requirements

This project was developed and tested using **Anaconda/Miniconda** on Python 3.13.  
The recommended way to install dependencies is from the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate furuta-rl
```

Note: The `pinocchio.casadi` module is only available in the `conda-forge` build of Pinocchio.
For full functionality, install via the provided `environment.yml` (do not use pip).

## Usage

Launch JupyterLab:

```bash
jupyter lab
```

Run all cells in the `Robotics_RL_FurutaPendulum.ipynb` notebook sequentially to visualise the results.

## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
