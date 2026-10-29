# On-Device Face Mask Compliance Detection

Real-time 3-state detection, running fully on the laptop (no cloud, no internet):
**MASK** (green) | **IMPROPER** (orange) | **NO MASK** (red)

Pipeline: Webcam -> Haar face detector -> crop -> TFLite MobileNetV2 -> label + violation count

## One-time setup
1. Install Python 3.10 or 3.11 (tick "Add Python to PATH" on Windows).
2. Windows: double-click `setup.bat`   |   Mac/Linux: `bash setup.sh`
3. Download the Kaggle dataset "Face Mask Detection" (andrewmvd) and unzip it into a
   folder named `raw_data` here (it must contain `images/` and `annotations/`).

## Build the model (once)
Open a terminal in this folder and activate the environment
(Windows: `venv\Scripts\activate`, Mac/Linux: `source venv/bin/activate`), then:

    python prepare_dataset.py raw_data
    python train.py
    python evaluate.py
    python convert_tflite.py

## Run the demo (any time)
Windows: double-click `run_demo.bat`   |   Mac/Linux: `bash run_demo.sh`
Press `q` to quit, `s` to save a screenshot.
