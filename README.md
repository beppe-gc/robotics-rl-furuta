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
Clone the repo and install dependencies:

```bash
git clone https://github.com/beppe-gc/robotics-rl-furuta.git
cd robotics-rl-project
conda env create -f environment.yml
conda activate furuta-rl

## Usage

Launch JupyterLab:

```bash
jupyter lab


