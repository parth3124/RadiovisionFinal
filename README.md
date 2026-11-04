# RadioVision Camera

### Camera-Supervised Human Position Estimation Using Wi-Fi CSI

RadioVision Camera is an experimental project that explores whether Wi-Fi Channel State Information (CSI) can be used to estimate the position and movement of a person inside a fixed area.

The project uses a camera to obtain ground-truth position coordinates during data collection. These coordinates are paired with CSI measurements to train machine-learning models that learn the relationship between changes in wireless signals and a person's physical location.

The long-term objective is to estimate human position and movement trajectories using Wi-Fi signals, without requiring a camera during inference.

---

## 1. Project Objective

Traditional camera-based tracking systems rely on visual information to locate people. RadioVision explores an alternative approach using wireless signals.

When a person moves through an area covered by Wi-Fi transmitters and receivers, their movement affects the wireless propagation environment. These changes are reflected in the Channel State Information (CSI) recorded by the receiver.

The central research question is:

**Can a machine-learning model learn to estimate a person's position from Wi-Fi CSI using camera-generated position labels as supervision?**

The project currently investigates two related prediction tasks:

- **Zone Classification:** Predict which predefined region of the experimental area a person occupies.
- **Position Regression:** Predict the person's continuous 2D position coordinates `(x, y)` within the experimental area.

---

## 2. Experimental Setup

The experiments are conducted in a fixed indoor area with approximate dimensions:

- Width: 1.00 m
- Depth: 1.28 m

The setup consists of:

- Wi-Fi CSI-capable ESP32 devices configured for transmission and reception.
- A Logitech C270 webcam for visual tracking.
- A computer for CSI collection, camera processing, calibration, and model training.

The camera observes the experimental area while the wireless devices capture CSI measurements.

The camera and CSI recordings are collected together so that wireless measurements can be associated with corresponding ground-truth position information.

---

## 3. System Architecture

The project follows a camera-supervised learning pipeline.

```text
                 EXPERIMENTAL AREA
                         |
              +----------+----------+
              |                     |
          Webcam              Wi-Fi Devices
              |                     |
       Camera Frames             CSI Data
              |                     |
       Person Detection       Signal Processing
              |                     |
       Position Tracking      CSI Feature Extraction
              |                     |
        Ground Truth                |
          (x, y)                    |
              |                     |
              +----------+----------+
                         |
                  Paired Dataset
                         |
                 Machine Learning
                         |
              +----------+----------+
              |                     |
        Zone Classification   Position Regression
              |                     |
          Zone Label             (x, y)
```

The camera provides supervision during training and evaluation. The intended inference pipeline uses CSI-derived features as model input, without requiring camera input.

---

## 4. Camera Calibration and Ground-Truth Generation

The camera is calibrated to map image coordinates to physical coordinates within the experimental area.

The calibration process uses the known physical dimensions and marked corners of the area to establish a mapping between camera coordinates and the real-world coordinate system.

The calibration script is:

```bash
python3 overhead_calibrate.py \
    --cam 0 \
    --area-w 1.00 \
    --area-d 1.28 \
    --out calib.json
```

The resulting calibration file is used during data collection to convert detected person positions into physical coordinates.

The ground-truth tracking data is stored in `track.npz`, containing:

| Variable | Description |
|---|---|
| `t` | Timestamp |
| `px`, `py` | Camera/image position coordinates |
| `x`, `y` | Position in the calibrated physical area |
| `visible` | Visibility status of the tracked person |

This tracking information provides the position labels used for supervised learning.

---

## 5. Data Collection

The collection pipeline records CSI measurements alongside camera-based position tracking.

The main collection script is:

```bash
python3 overhead_collect.py
```

The collection pipeline supports the following experimental labels:

| Label | Description |
|---|---|
| `walk_free` | Person walking freely through the experimental area |
| `stop_and_go` | Person alternating between movement and stationary periods |
| `empty` | Experimental area without a person |

The collection script supports parameters for:

- Serial ports
- Receiver names
- Baud rate
- Session identification
- Recording duration
- Voice prompts
- Collection plans

Example collection arguments:

```bash
python3 overhead_collect.py \
    --ports <SERIAL_PORTS> \
    --names <DEVICE_NAMES> \
    --session session1 \
    --out dataset_over \
    --duration 90 \
    --plan
```

The exact number and configuration of receiver devices depend on the experiment.

Each recording produces CSI data and camera-derived tracking information.

---

## 6. Feature Extraction

Raw CSI measurements contain information about changes in the wireless channel.

The project investigates different feature representations to understand which characteristics of the CSI signal are useful for position estimation.

The evaluated feature families include:

- **M0:** Statistical signal features, including mean and standard deviation of log-power and RSSI-related measurements.
- **M1:** Per-subcarrier mean and standard deviation of packet-normalised log-amplitude.
- **M1G:** Combination of M0 and M1 features.

These representations convert CSI measurements into numerical feature vectors that can be used by conventional machine-learning models.

---

## 7. Machine Learning Models

Two prediction tasks are investigated.

### A. Zone Classification

The objective is to classify the person's location into predefined regions of the experimental area.

The documented baseline uses:

- StandardScaler
- Logistic Regression
- Balanced class weights

The classifier learns a mapping from CSI feature vectors to discrete spatial zones.

### B. Position Regression

Instead of predicting a discrete zone, the regression model estimates continuous physical coordinates.

The documented baseline uses:

- StandardScaler
- Ridge Regression

The model predicts:

\[
\hat{y} = (\hat{x}, \hat{y})
\]

where the predicted coordinates represent the estimated physical position of the person.

The objective is to minimise the difference between the predicted coordinates and the camera-derived ground truth.

---

## 8. Evaluation

The system is evaluated using both classification and regression metrics.

**Zone Classification**
- Accuracy
- Balanced Accuracy
- Confusion Matrix

**Position Regression**
- Mean Euclidean Position Error
- Comparison against a mean-position baseline
- Cross-validation across recording sessions

The Euclidean position error is:

\[
e_i = \sqrt{(x_i-\hat{x}_i)^2+(y_i-\hat{y}_i)^2}
\]

The mean position error is:

\[
E = \frac{1}{N}\sum_{i=1}^{N}e_i
\]

Cross-session evaluation is particularly important because a model should ideally generalise to recordings collected in sessions that were not used during training.

---

## 9. Repository Structure

The main scripts in the project include:

| File | Purpose |
|---|---|
| `overhead_calibrate.py` | Camera calibration |
| `overhead_collect.py` | Synchronized CSI and camera data collection |
| `overhead_common.py` | Shared utilities |
| `overhead_eval.py` | Model evaluation |
| `multi_collect.py` | Multi-device data collection |

Additional experiment-specific scripts and generated datasets may be present in the repository.

---

## 10. Current Research Status

RadioVision Camera is an ongoing experimental project.

The current work focuses on:

1. Reliable camera calibration and physical coordinate tracking.
2. Synchronized collection of CSI and camera-derived ground truth.
3. Extracting useful statistical representations from CSI.
4. Comparing zone classification and continuous position regression.
5. Investigating generalisation across different recording sessions.
6. Improving the accuracy and stability of CSI-based position estimation.

Preliminary experiments have shown that the prediction task is challenging, particularly when evaluating generalisation across sessions.

The current results should be treated as experimental baselines rather than evidence of reliable real-world tracking.

---

## 11. Future Work

Potential directions for further development include:

- Improving CSI feature representations.
- Investigating temporal models that use sequences of CSI measurements.
- Evaluating multiple spatially distributed receivers.
- Improving cross-session generalisation.
- Reducing position estimation error.
- Reconstructing continuous human movement trajectories from CSI predictions.

The ultimate goal is to develop a low-cost wireless sensing system capable of estimating human movement and position without relying on camera input during deployment.

---

## Research Direction

RadioVision Camera investigates the possibility of using commodity Wi-Fi signals as a sensing modality for indoor human positioning.

By combining camera-supervised ground-truth generation, CSI feature extraction, and machine learning, the project aims to explore how much spatial information can be recovered from changes in wireless channel measurements.
