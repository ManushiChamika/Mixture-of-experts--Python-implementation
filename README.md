# Mixture of Experts (MoE) Visualization Tool 🎯

> Created by NotebookLM - Empowering AI Research and Development

## 🎥 Video Tutorial: Introduction on MoE

https://github.com/ManushiChamika/Mixture-of-experts--Python-implementation/tree/main/MoE-intro-video

[Click here to download and watch the full tutorial video](MoE-intro-video/moe_implementation.mp4)

## What is this project about?

This project provides visualization tools for understanding Mixture of Experts (MoE) models - a type of neural network architecture where multiple "expert" networks specialize in different inputs. Think of it as having multiple specialist doctors in a hospital: each expert handles specific types of cases, and a "router" directs patients to the most appropriate specialist.

### Why is this important?
- **Understanding Model Behavior**: See how your MoE model distributes work among experts
- **Detecting Problems**: Quickly identify issues like expert collapse (where one expert takes all the work)
- **Optimizing Performance**: Monitor routing decisions and expert utilization to improve model efficiency

### What does it do?
1. Takes training logs from your MoE model
2. Creates visual diagnostics (heatmaps, graphs) showing expert usage
3. Helps you understand if your model is learning effectively

Perfect for researchers, students, and practitioners working with MoE architectures who want to visualize and understand their model's behavior.

## 📊 Key Features

- **Router Affinity Visualization**: Heatmap visualization of how inputs are routed to different experts
- **Expert Utilization Tracking**: Time-series plots showing how expert usage evolves during training
- **Training Metrics**: Monitor router entropy and routing accuracy over epochs
- **Interactive Visualizations**: Clear, publication-quality plots using matplotlib and seaborn

## 🚀 Quick Start

### Prerequisites

- Python 3.x
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd python-moe
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install numpy pandas matplotlib seaborn
```

### Usage

Run the visualization script:
```bash
python visualize_outputs.py
```

This will generate several visualization files in the `moe_outputs` directory:
- `router_affinity_matrix_heatmap.png`: Router's routing decisions
- `expert_utilization.png`: Expert usage over time
- `router_entropy.png`: Router's decision entropy
- `routing_accuracy.png`: Routing accuracy metrics

## 📈 Understanding the Outputs

### Router Affinity Matrix
- **What it shows**: How strongly each input is routed to different experts
- **Interpretation**: Brighter colors indicate stronger routing affinities
- **Good signs**: Clear patterns of specialization, balanced expert usage

### Expert Utilization
- **What it shows**: Percentage of inputs routed to each expert over training
- **Interpretation**: Lines show how expert usage evolves
- **Warning signs**: 
  - Single expert near 100% (collapse)
  - Many experts near 0% (underutilization)

### Router Entropy
- **What it shows**: Uncertainty in router decisions
- **Expected trend**: Should generally decrease as training progresses
- **Interpretation**: Lower values indicate more confident routing

### Routing Accuracy
- **What it shows**: How often router picks the "correct" expert
- **Expected trend**: Should increase during training
- **Target**: Higher values indicate better routing decisions

## 🔧 Troubleshooting

### Common Issues

1. **Blank Plots**
   - Ensure training log has valid numeric data
   - Check for proper array shapes
   - Verify epoch numbers are monotonically increasing

2. **Missing Data**
   ```python
   # Quick diagnostic check
   import json, numpy as np
   with open('moe_outputs/training_log.txt') as f:
       log = [json.loads(l) for l in f]
   print('epochs:', [e['epoch'] for e in log])
   print('util shape:', np.array([e['utilization'] for e in log]).shape)
   ```

3. **Style Issues**
   - Use `sns.set()` for consistent styling
   - Check matplotlib backend compatibility

## 📁 Project Structure

```
python-moe/
├── visualize_outputs.py     # Main visualization script
├── moe_outputs/
│   ├── training_log.txt        # Training metrics and logs
│   ├── router_affinity_matrix.csv  # Router decisions
│   └── [generated plots]       # Output visualizations
└── README.md
```

## 📊 Training Log Format

Each line in `training_log.txt` contains a JSON object with the following structure:
```json
{
    "epoch": 1,
    "routing_acc": 0.5841,
    "entropy": 1.4012,
    "utilization": [0.2257, 0.1031, 0.0735, 0.2797, 0.1871, 0.1309],
    "metrics": {
        "accuracy": 0.4356,
        "loss": 2.5902
    }
}
```

## 🎓 Further Reading

- [Mixture of Experts Paper](https://arxiv.org/abs/1701.06538)
- [Understanding MoE Visualizations](https://arxiv.org/abs/2202.08906)
- [Router Design Patterns](https://arxiv.org/abs/2204.09424)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Thanks to all contributors and researchers in the MoE field
- Visualization inspiration from various MoE papers and implementations

---
Made with ❤️ for the ML community