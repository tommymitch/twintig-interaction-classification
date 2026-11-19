# twintig-interaction-classification

A small 1D CNN that classifies hand gestures from accelerometer data recorded
with a Twintig sensor. There are six classes: taps on three silicone surfaces of
different hardness (soft, medium and hard), and three mid-air movements (a tap,
an upward swipe, and random hand movement). `train_kfold.py` trains the model
and evaluates it with 5-fold stratified cross-validation.

## Hardware and data capture
The data was recorded with a **Twintig Carpus** inertial sensor connected over
USB, logged in the x-IMU3 CSV format (timestamp, 3-axis gyroscope, 3-axis
accelerometer) at about **32.26 kHz**.

Each row of the dataset is one gesture, extracted from those recordings:
- The signal is the **accelerometer magnitude**, √(x² + y² + z²), in g.
- For the tap classes (0–4), a tap is detected when the magnitude goes above
  2 g. The window runs from 50 ms before that point to 100 ms after it: 150 ms,
  which is 4837 samples. Detections are at least 500 ms apart.
- For class 5 (random hand movement) there is no tap to detect, so 24 windows of
  the same length are taken at evenly spaced points through each recording.

The script uses the first 4836 samples of each row, so that the length divides
evenly through the model's two pooling layers.

The extraction was done by `OpenAndWrite.py` and `OpenAndWriteNoTaps.py` in the
`18CaptureAndAnalyse` folder of the wider project.

## Files
- `train_kfold.py` – model, training, and evaluation
- `numbersSoftMediumHardBigAirTapsBiggerTapupAir.csv` – dataset (432 rows, 72 per class)
  - column 0: class label (0–5)
  - remaining 4837 columns: accelerometer magnitude (g), one 150 ms window
- `requirements.txt` – package versions used

## Running
```
pip install -r requirements.txt
python train_kfold.py
```
It uses CUDA if available, otherwise the CPU. Plots open in windows
(close each one to continue).

## Classes
| Label | Class |
|---|---|
| 0 | Tap on soft silicone |
| 1 | Tap on medium silicone |
| 2 | Tap on hard silicone |
| 3 | AirTap (tapping in mid-air) |
| 4 | TapUp (mid-air upward swipe) |
| 5 | Air (hand moving randomly in free space) |
