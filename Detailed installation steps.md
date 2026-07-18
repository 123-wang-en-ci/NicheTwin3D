

# NicheTwin3D Installation Guide

This installation guide essentially requires you to do four things: Clone the GitHub repository to extract the `Server` directory, download `data.zip` and place the models/data into the `Server` directory, create a Conda environment to run `server.py`, and finally run the Unity frontend application.

---

## I. Unified Installation Directory

It is highly recommended to use a fixed directory:

```text
D:\NicheTwin3D
```

Ultimately, we want to achieve this folder structure (**where the frontend and backend folders are at the same level**):

```text
D:\NicheTwin3D
├─ Server        ← Python Backend Directory
├─ NicheTwin3D   ← Windows Frontend Application
│  ├─ NicheTwin3D.exe
│  └─ NicheTwin3D_Data
│     └─ StreamingAssets
└─ Downloads     ← Temporary folder for data.zip and NicheTwin3D_Windows.zip
```

**Avoid placing it in:**

- `C:\Program Files`
- Paths containing non-English characters
- Paths with multiple spaces

*These paths often cause permission, path-parsing, or encoding issues.*

---

## II. Install Prerequisites

### 1. Install Git for Windows
Git is used to download the GitHub source code.

**Steps:**
1. Go to the official Git for Windows installation page: [git-scm.com](https://git-scm.com/install/windows).
2. Download the `64-bit Git for Windows Setup`.
3. Double-click the installer and proceed with the default settings by continuously clicking "Next".

### 2. Install Anaconda (or Miniconda)
For beginners, **Anaconda** is recommended because it provides the **Anaconda Prompt** in the start menu, which makes subsequent operations much easier. You can download the Windows 64-bit graphical installer from [anaconda.com](https://www.anaconda.com/download/success).

**Recommended installation settings:**
- Install for: **Just Me**
- Add Anaconda to PATH: **Uncheck**
- Register Anaconda as default Python: **Check**

After installation, open your Windows Start Menu and search for:
```text
Anaconda Prompt
```
**Note:** All subsequent command-line operations must be executed within **Anaconda Prompt**. Do not use standard CMD or PowerShell.

---

## III. Create the Root Folders

Open **Anaconda Prompt** and run the following commands line by line (press `Enter` after each line):

```bat
D:
cd \
mkdir NicheTwin3D
cd /d D:\NicheTwin3D
mkdir Downloads
```

Verify the creation by running:
```bat
dir
```
You should see the `Downloads` folder listed.

---

## IV. Download GitHub Source & Extract Backend

Continue in the **Anaconda Prompt** and enter:

```bat
cd /d D:\NicheTwin3D
git clone https://github.com/123-wang-en-ci/NicheTwin3D.git temp_src
move temp_src\Server D:\NicheTwin3D\Server
rmdir /s /q temp_src
```
*(These commands clone the repository, extract only the `Server` folder to `D:\NicheTwin3D`, and delete the rest of the unnecessary source files.)*

Check if the backend directory was extracted successfully:
```bat
dir D:\NicheTwin3D\Server
```
You should see files like `server.py`, `environment.yml`, `model_engine.py`, etc.

*If you see an error like `'git' is not recognized as an internal or external command`, it means Git is not installed properly, or you haven't restarted Anaconda Prompt after installing Git.*

---

## V. Download the frontend and data compression package

This step is crucial. The GitHub source code does not include model weights, H5AD data, or front-end software. You need to download it from ([huggingface.co](https://huggingface.co/datasets/www123222/NicheTwin3D/tree/main)). On the Hugging Face page, you can see 'NicheTwin3D Windows.zip' and ' data.zip' are two key files, where 'data.zip' is the data/model package, and 'NicheTwin3D Windows.zip' is the front-end software compressed package.

**Steps:**
1. Open the NicheTwin3D Hugging Face repository: [huggingface.co/datasets/www123222/NicheTwin3D/tree/main](https://huggingface.co/datasets/www123222/NicheTwin3D/tree/main).
2. Download `data.zip`.
3. Save it to: `D:\NicheTwin3D\Downloads`
4. Once downloaded, right-click `data.zip` and select **Extract All...**
5. Extract it to: `D:\NicheTwin3D\Downloads\data`

---

## VI. Extract the Windows Frontend App

For the 'NicheTwin3D_Windows.zip' Windows frontend package, do not double-click exe directly inside the compressed package; you must extract it first.

**Steps:**

1. Locate `NicheTwin3D_Windows.zip` in your File Explorer.

2. Right-click the file and select **Extract All...**

3. Choose the extraction destination:

   ```text
   D:\NicheTwin3D
   ```

4. After extraction, ensure this executable exists:

   ```text
   D:\NicheTwin3D\NicheTwin3D\NicheTwin3D.exe
   ```

5. And ensure this directory exists:

   ```text
   D:\NicheTwin3D\NicheTwin3D\NicheTwin3D_Data\StreamingAssets
   ```

   *(The ZIP package may already contain a `StreamingAssets` folder and a `unity_cell_data.csv`. You can ignore it, as our latest backend will automatically locate and overwrite it.)*

---

## VII. Move Data Files to the Server Directory

Open this directory:
```text
D:\NicheTwin3D\Downloads\data
```
Check for these specific files:
- `Allen2022Molecular_lps_MsBrainAgingSpatialDonor_14_1.h5ad`
- `nicheformer_weights.pth`

Copy or move these extracted model (`.pth`) and data (`.h5ad`) files directly into:
```text
D:\NicheTwin3D\Server
```

**Your final folder structure must look exactly like this:**
```text
D:\NicheTwin3D\Server\server.py
D:\NicheTwin3D\Server\environment.yml
D:\NicheTwin3D\Server\Allen2022Molecular_lps_MsBrainAgingSpatialDonor_14_1.h5ad
D:\NicheTwin3D\Server\nicheformer_weights.pth
```

**Avoid nested paths like:**
- ❌ `D:\NicheTwin3D\Server\data\Allen2022Molecular_lps_MsBrainAgingSpatialDonor_14_1.h5ad`

Verify using Anaconda Prompt:
```bat
dir /b D:\NicheTwin3D\Server\*.h5ad
dir /b D:\NicheTwin3D\Server\*.pth
```
You should see the `.h5ad` and `.pth` files listed.

---

## VIII. Enter the Backend Directory

Open **Anaconda Prompt** and type:
```bat
cd /d D:\NicheTwin3D\Server
```
Verify your current path:
```bat
cd
```
It should print: `D:\NicheTwin3D\Server`. This step is crucial, as all subsequent Python commands must be executed within this directory.

---

## IX. Create the Conda Environment

In **Anaconda Prompt**, run:
```bat
conda create -n aivc python=3.9.23 -y
```
Once completed, activate the environment:
```bat
conda activate aivc
```
After activation, your command line prefix should change to `(aivc)`, e.g., `(aivc) D:\NicheTwin3D\Server>`.

Check your Python version:
```bat
python --version
```
It should display `Python 3.9.23`.

---

## X. Install PyTorch

Ensure your command line starts with `(aivc)`.

First, upgrade pip:
```bat
python -m pip install --upgrade pip
```
Then, install PyTorch (CUDA 11.8 version):
```bat
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchtext==0.16.0+cpu --extra-index-url https://download.pytorch.org/whl/cu118
```

After installation, verify PyTorch:
```bat
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```
The ideal output should be:
```text
2.1.0+cu118
True
```
*(Meaning: PyTorch CUDA 11.8 is successfully installed, and a compatible NVIDIA GPU was detected.)*

---

## XI. Install Dependencies from `environment.yml`

Ensure you are still in `D:\NicheTwin3D\Server` and the `(aivc)` environment is active. Run:
```bat
conda env update -n aivc -f environment.yml
```
*(Lots of downloading and installing messages will appear, which is normal.)*

Once finished, run a basic test:
```bat
python -c "import fastapi, uvicorn, scanpy, pandas, numpy, torch; print('basic imports OK')"
```
If it outputs `basic imports OK`, the core dependencies are successfully installed.

---

## XII. Start the Python Backend

Thanks to the backend's auto-path-detection feature, **you do not need to input any complex path parameters**. As long as the frontend and backend folders are placed side-by-side, it will automatically locate the frontend's data folder.

Simply execute:
```bat
python server.py
```

Upon successful startup, you should see logs similar to this:
```text
[System] Initialize Nicheformer engine.....
[LifeSpan] Server starting up...
[Backend] Loading data: Allen2022Molecular_lps_MsBrainAgingSpatialDonor_14_1.h5ad .....
[Sync] Generate CSV for Unity ...
[Successfully] CSV saved to: D:\NicheTwin3D\NicheTwin3D\NicheTwin3D_Data\StreamingAssets\unity_cell_data.csv
Uvicorn running on http://127.0.0.1:8000
```

**IMPORTANT:** **Do not close this command window.** This is the backend server. If you close it, any AI-related features in the Unity frontend will fail.

---

## XIII. Verify CSV Generation

Keep the backend window running. Open a new **Anaconda Prompt** or use Windows File Explorer.

Check if this file exists and has the latest modification date:
```text
D:\NicheTwin3D\NicheTwin3D\NicheTwin3D_Data\StreamingAssets\unity_cell_data.csv
```
As long as you can see the file size and the latest timestamp, the backend has successfully synced the CSV to the frontend's location!

---

## XIV. Confirm the Backend is Alive

With the backend window still open, open your web browser and visit:
```text
http://127.0.0.1:8000/docs
```
If you see the FastAPI / Swagger API documentation page, the backend is successfully running.

---

## XV. Start the Unity Frontend

Keep the backend window running. Open your Windows File Explorer and navigate to:
```text
D:\NicheTwin3D\NicheTwin3D
```
Double-click:
```text
NicheTwin3D.exe
```
Upon startup, the Unity frontend will load the `unity_cell_data.csv` generated by the backend. Certain interactive features will send requests to `http://127.0.0.1:8000`.

**The complete running state should be:**
- **Window 1:** Anaconda Prompt running `python server.py` (Must stay open)
- **Window 2:** `NicheTwin3D.exe` (The visual frontend interface)

---

## XVI. Daily Routine (After First-Time Installation)

You only need to install the dependencies once. For future use, follow this sequence:

### 1. Start the Backend
Open **Anaconda Prompt**:
```bat
cd /d D:\NicheTwin3D\Server
conda activate aivc
python server.py
```
Wait until you see it running on `127.0.0.1:8000`, and **do not close the window**.

### 2. Start the Frontend
Double-click the executable:
```text
D:\NicheTwin3D\NicheTwin3D\NicheTwin3D.exe
```

### 3. Shutting Down
1. Close the Unity frontend window.
2. Go back to the Anaconda Prompt, press `Ctrl + C`. If prompted to terminate the batch job, type `Y` and press `Enter`.

---

## XVII. TL;DR - Complete Command Summary

Assuming Git and Anaconda are installed, and the `.zip` files are in place, here is the full copy-paste command script to set everything up:

```bat
D:
cd \
mkdir NicheTwin3D
cd /d D:\NicheTwin3D
mkdir Downloads

git clone https://github.com/123-wang-en-ci/NicheTwin3D.git temp_src
move temp_src\Server D:\NicheTwin3D\Server
rmdir /s /q temp_src

cd /d D:\NicheTwin3D\Server

conda create -n aivc python=3.9.23 -y
conda activate aivc

python -m pip install --upgrade pip
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchtext==0.16.0+cpu --extra-index-url https://download.pytorch.org/whl/cu118
conda env update -n aivc -f environment.yml

python -c "import fastapi, uvicorn, scanpy, pandas, numpy, torch; print('basic imports OK')"

python server.py
```
