# RadioVision Seq1

### Sequential Wi-Fi CSI-Based Indoor Environment Classification

RadioVision Seq1 is an experimental project that investigates whether changes in Wi-Fi Channel State Information (CSI) over time can be used to distinguish between different physical conditions in an indoor environment.

Using ESP32 devices configured for Wi-Fi transmission and CSI reception, the project collects wireless channel measurements under different environmental conditions and applies machine-learning techniques to classify the observed environment.

The project focuses on extracting meaningful temporal and statistical representations from CSI measurements and investigating their usefulness for indoor wireless sensing.

---

## 1. Project Objective

The project investigates three environmental classes:

| Class | Description |
|---|---|
| `empty` | No person or target object in the monitored environment |
| `object` | An object is present in the monitored environment |
| `person` | A person is present in the monitored environment |

The dataset also contains spatially labelled recordings:

- `obj_Z1`, `obj_Z2`, `obj_Z3`
- `person_Z1`, `person_Z2`, `person_Z3`

These recordings allow experiments involving the presence and spatial arrangement of objects and people.

The central research question is:

**Can temporal patterns in Wi-Fi CSI reveal information about changes in an indoor environment, enabling machine-learning-based environmental classification?**

---

## 2. Experimental Setup

The experimental setup uses two ESP32 devices:

- **Transmitter:** Generates Wi-Fi packets.
- **Receiver:** Captures Channel State Information corresponding to transmitted packets.

CSI describes how a wireless signal is affected by its propagation environment.

When a person or object enters the monitored area, the wireless propagation paths can change due to reflection, absorption, and scattering. These changes are reflected in the recorded CSI measurements.

The project investigates whether these signal variations can be used to identify different environmental conditions.

---

## 3. System Architecture

The project follows a sequential CSI classification pipeline.

```text
       TWO-ESP WI-FI EXPERIMENTAL SETUP
                       |
                Wi-Fi Transmission
                       |
                CSI Measurements
                       |
                 Data Collection
                       |
              Sequential CSI Dataset
                       |
                Signal Processing
                       |
               Feature Extraction
                       |
              Temporal Feature Sets
                       |
               Machine Learning
                       |
             Environmental Classification
                       |
           +-----------+-----------+
           |           |           |
         Empty       Object      Person
```

The model uses CSI-derived features as input, without requiring camera images.

The experiments investigate whether temporal information provides useful representations of changes in the wireless propagation environment.

---

## 4. Dataset Organization

The experimental dataset is organised under `dataset_room`.

The recorded conditions include:

```text
dataset_room/
│
├── empty/
│
├── obj_Z1/
├── obj_Z2/
├── obj_Z3/
│
├── person_Z1/
├── person_Z2/
└── person_Z3/
```

The project also uses multiple recording sessions:

| Session | Description |
|---|---|
| `r01` | First recording session |
| `r02` | Second recording session |
| `r03` | Third recording session |

CSI recordings are stored in NumPy archive files, including:

- `csi_link.npz`
- `csi_rx1.npz`

Separating recordings by session enables evaluation across independently collected experimental sessions.

---

## 5. Feature Engineering

A major component of RadioVision Seq1 is investigating different representations of sequential CSI measurements.

Several feature configurations were developed and compared.

### A — Statistical Amplitude Features

The baseline representation uses statistical characteristics of CSI amplitude measurements.

It provides a reference for evaluating more elaborate temporal and spectral representations.

### B1 — Amplitude with Temporal Differences

B1 extends the baseline representation with temporal difference information.

The objective is to capture changes between successive CSI measurements rather than relying exclusively on statistical summaries.

### B2 — Temporal Dynamics

B2 focuses on temporal characteristics of CSI measurements.

It investigates whether the evolution of wireless measurements over time contains useful information for distinguishing environmental conditions.

### B3 — Band and Spectral Features

B3 investigates frequency-related characteristics of CSI measurements.

The objective is to capture additional signal structure that may not be represented by simple statistical features.

### C — Additional Feature Representation

Configuration C was introduced in subsequent experiments to investigate another CSI feature representation.

Its implementation is maintained separately in the experiment code.

---

## 6. Model Training and Evaluation

The project includes feature extraction, temporal training and evaluation, signal analysis, phase checks, and zone-related experiments.

A key aspect of the evaluation is testing generalisation across different recording sessions:

- `r01`
- `r02`
- `r03`

Session-based evaluation is important because neighbouring CSI windows from the same recording can be highly correlated.

Testing on independently collected sessions provides a more meaningful assessment of whether a model learns environmental characteristics rather than session-specific signal patterns.

The experiments also investigate:

- Effects of packet normalisation.
- Differences between statistical and temporal feature representations.
- Class-wise prediction behaviour.
- Generalisation across recording sessions.
- The influence of CSI signal characteristics on classification.

---

## 7. Research Challenges

The experiments explore several challenges associated with CSI-based environmental sensing.

**Session generalisation**

Wireless measurements can vary between recording sessions. Models must learn meaningful environmental characteristics rather than session-specific variations.

**Temporal feature selection**

Different temporal and spectral representations capture different aspects of the wireless channel. Identifying the most informative features is an important part of the project.

**Class discrimination**

Different environmental conditions can produce overlapping CSI patterns, making reliable classification challenging.

**Limited experimental data**

The diversity and consistency of real CSI recordings influence the ability of models to generalise to new conditions.

---

## 8. Future Work

Potential directions for further development include:

- Investigating the effect of environmental and session-dependent CSI variations.
- Improving the consistency of data collection.
- Exploring temporal neural networks for sequential CSI classification.
- Comparing statistical, temporal, and spectral feature representations.
- Evaluating models on entirely new physical recordings.
- Investigating whether additional spatially distributed receivers improve environmental sensing.
- Extending classification towards more detailed spatial understanding of the monitored environment.

---

## 9. Repository Contents

The repository contains scripts and utilities for:

- CSI data collection and preprocessing.
- Feature extraction and comparison.
- Temporal model training and evaluation.
- CSI signal and phase analysis.
- Dataset and feature audits.
- Zone-related evaluation experiments.

The dataset and experiment outputs are organised separately from the processing and evaluation scripts.

---

## Research Direction

RadioVision Seq1 explores the use of commodity Wi-Fi Channel State Information as a sensing modality for indoor environmental classification.

By investigating statistical, temporal, and spectral representations of wireless channel measurements, the project aims to understand how changes in the surrounding environment can be inferred from Wi-Fi signals.

The broader objective is to explore low-cost wireless sensing approaches for detecting people and objects without relying on conventional visual sensing systems.
