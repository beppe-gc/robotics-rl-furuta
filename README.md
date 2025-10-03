# Robotics RL Project – Furuta Pendulum Swing-Up with PPO

This project applies **Reinforcement Learning (RL)** to control a **Furuta pendulum** (torque-driven inverted pendulum on a rotating base). The task is to swing up the pendulum and balance it in the upright position using **Proximal Policy Optimization (PPO)**.

## Project Structure
- `Robotics_RL_FurutaPendulum.ipynb` – main Jupyter notebook with experiments and results  
- `src/` – source code
  - `envs/` – custom Gymnasium environments for the Furuta pendulum  
  - `agents/` – RL agents (PPO implementation)  
  - `utils/` – helper scripts for dynamics and training  
  - `pendulum_description/` – system model and description files  
- `docs/` – report and original project description  
- `data/` – (optional) logs or datasets  

## Setup
Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/robotics-rl-project.git
cd robotics-rl-project
conda env create -f environment.yml
conda activate furuta-rl
