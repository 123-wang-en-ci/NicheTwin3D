# NicheTwin3D: A universal digital twin paradigm for virtual cell initiative

[![Unity](https://img.shields.io/badge/Unity-2021.3%2B-blue.svg)](https://unity.com/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-Nicheformer-orange.svg)](https://huggingface.co/datasets/www123222/NicheTwin3D)

This system is an interactive spatial transcriptomics digital twin framework based on Unity and deep learning. It combines a high-performance 3D rendering engine with advanced AI models (Nicheformer) to provide researchers with an intuitive platform for exploring cell distribution, gene expression, cell type annotation, and tissue regional segmentation.

<img width="1800" height="2184" alt="abcd" src="https://github.com/user-attachments/assets/bf473163-7d06-42a2-83ec-9671ad49bbe3" />



Project video explanation: [NicheTwin3D](https://www.youtube.com/watch?v=qL38RVwW2h8)

---

## 🛠️ Installation

### 1. Clone Repository & Environment Preparation

```bash
git clone https://github.com/your-username/NicheTwin3D.git
cd NicheTwin3D
```

### 2. Download Weights & Data (Essential Weights & Data)

**⚠️ IMPORTANT**: Due to large file sizes, model weights and H5AD datasets are NOT included directly in the GitHub repository. Please download them from Hugging Face:

- **Download Link**: [Hugging Face - NicheTwin3D Dataset](https://huggingface.co/datasets/www123222/NicheTwin3D/tree/main)
- **File Placement**:
  - Place `.h5ad` data files in the `Assets/Scripts/Server/` directory.
  - Place `nicheformer_weights.pth` weight files in the `Assets/Scripts/Server/` directory.

### 3. Python Backend Configuration

```bash
# Enter the backend directory
cd Assets/Scripts/Server

# Install dependencies
pip install -r requirements.txt

# Start the service
python server.py
```

*The backend will run at `http://127.0.0.1:8000` after a successful start.*

### 4. Unity Frontend Configuration

1. Directly run **NicheTwin3D.exe**.
2. Ensure that `unity_cell_data.csv` exists in the `StreamingAssets` directory (if it is missing, run the backend and click **Sync** to generate it).

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](https://github.com/123-wang-en-ci/NicheTwin3D/blob/main/LICENSE) file for details. SPDX-License-Identifier: BSD-3-Clause
