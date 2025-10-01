# 🌬️ Project ATLAS
### Automated Turbine Layout And Siting

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)](https://www.tensorflow.org/)

> An AI-powered solution that uses deep reinforcement learning to discover optimal wind turbine layouts, maximizing energy production by intelligently minimizing wake interference.

---

## 🎯 The Challenge

Wind turbines create "wakes" of slower, turbulent air behind them, reducing downstream turbine efficiency by **10-15%**. Traditional layout design relies on manual iteration or simple grid patterns, leaving significant energy potential untapped.

**Project ATLAS solves this using AI.**

## ✨ Key Features

- 🤖 **Deep Reinforcement Learning** - DDPG agent learns optimal placement strategies
- 🌍 **Real-World Data** - Integrates actual terrain and wind climate data from Tamil Nadu, India
- ⚡ **Physics-Based Simulation** - Implements Jensen Wake Model for accurate AEP calculations
- 📊 **Proven Results** - Achieves 92.6-99.5% efficiency across different scenarios
- ⚙️ **Highly Configurable** - Simple YAML-based configuration for all parameters
- 📁 **Reproducible** - Automatic experiment tracking and result archiving

## 🚀 Quick Start

### Prerequisites

- Anaconda
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/realyashagarwal/Project-ATLAS.git
cd Project-ATLAS

# Create and activate the conda environment
conda env create -f environment.yml
conda activate atlas_env
```

### Run Your First Optimization

```bash
# Run with default settings (10 turbines, 500 episodes)
python src/train.py
```

Results will be automatically saved in `outputs/` with timestamp.

### Customize Your Experiment

Edit `config.yaml` to adjust parameters:

```yaml
turbine:
  max_turbines: 20  # Number of turbines to optimize

training:
  total_episodes: 1000  # Training duration
  
site:
  area_size: 2000  # Site dimensions in meters
```

## 📊 Results & Performance

| Configuration | Turbines Placed | AEP (MWh/year) | Efficiency |
|---------------|-----------------|----------------|------------|
| 10-Turbine    | 10              | 216,741        | **99.5%**  |
| 20-Turbine    | 19*             | 403,459        | **92.6%**  |

\*Agent learned that 19 turbines was more efficient than placing all 20

## 🏗️ Project Structure

```
ATLAS/
├── config.yaml              # Central configuration file
├── environment.yml          # Conda environment specification
├── README.md               # This file
├── REPORT.md               # Detailed technical report
│
├── data/
│   ├── terrain.tif         # SRTM elevation data
│   └── wind_data.lib       # Global Wind Atlas wind data
│
├── notebooks/
│   └── 1-Data_Exploration.ipynb  # Data parsing & visualization
│
├── outputs/                # Timestamped experiment results
│   └── 2025-10-01_21-24-00/
│       ├── config.yaml
│       ├── final_layout.png
│       └── training_log.txt
│
└── src/
    ├── agent.py            # DDPG agent implementation
    ├── environment.py      # Wind farm simulation environment
    ├── main.py            # Component testing script
    └── train.py           # Main training pipeline
```

## 🧠 How It Works

### 1. The Environment
A digital twin of a wind farm site incorporating:
- Real wind data from Global Wind Atlas (12 directional sectors)
- Terrain elevation from NASA SRTM
- Jensen Wake Model for physics-based AEP calculation

### 2. The Agent
A Deep Deterministic Policy Gradient (DDPG) agent that:
- Places turbines sequentially on the site
- Receives rewards for increasing AEP
- Gets penalties for constraint violations (spacing, boundaries)
- Learns optimal placement strategies through trial and error

### 3. The Training Process
Over hundreds of episodes, the agent:
- Explores different placement strategies
- Learns to avoid wake interference
- Discovers counter-intuitive but efficient layouts
- Converges to near-optimal solutions

## 💻 Technology Stack

**Core Technologies:**
- Python 3.10+
- TensorFlow 2.x (Deep Learning)
- Gymnasium (RL Environment)

**Data & Computation:**
- NumPy & Pandas (Data processing)
- Rasterio (Geospatial analysis)
- PyYAML (Configuration)

**Visualization:**
- Matplotlib (Plots)
- Tableau (Dashboard)

## 📚 Documentation

- **[REPORT.md](REPORT.md)** - Comprehensive technical documentation
- **[Data Exploration Notebook](notebooks/1-Data_Exploration.ipynb)** - Data parsing and analysis

## 🗺️ Data Sources

This project uses open data from:

- **Terrain:** NASA Shuttle Radar Topography Mission (SRTMGL1) via [OpenTopography](https://opentopography.org/)
- **Wind Climate:** [Global Wind Atlas](https://globalwindatlas.info/) by DTU & World Bank Group
- **Renewable Stats:** [Ministry of New and Renewable Energy (MNRE)](https://mnre.gov.in/), Government of India

## 🎓 Context

Developed as part of the **1M1B Green Internship**, this project demonstrates how advanced AI can address real-world climate challenges. Tamil Nadu has significant untapped wind energy potential—ATLAS aims to help maximize that potential through intelligent design.

## 🔮 Future Roadmap

- [ ] Terrain-aware optimization (slope penalties)
- [ ] Interactive Streamlit web interface
- [ ] Additional RL algorithms (PPO, SAC)
- [ ] Docker containerization
- [ ] Multi-turbine type support
- [ ] Real-time wind data integration

## 🤝 Contributing

Contributions are welcome! Whether it's bug fixes, feature additions, or documentation improvements, feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Yash Agarwal**
- GitHub: [@realyashagarwal](https://github.com/realyashagarwal)

## 🙏 Acknowledgments

Special thanks to:
- Technical University of Denmark for the Global Wind Atlas
- NASA and OpenTopography for terrain data
- The open-source community for the amazing tools and libraries

---

**If you find this project useful, please consider giving it a ⭐️**

*Made with ❤️ for a sustainable future*