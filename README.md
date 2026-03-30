# Analysis-of-growth-of-multicellular-tumour-spheroids-by-mathematical-models
# Tumor Growth Modeling using Verhulst, Montroll & PINN

## 📌 Overview
This project implements mathematical and machine learning models to analyze tumor growth dynamics using real experimental data from Chinese hamster V79 fibroblast tumor cells.

The study compares:
- Verhulst (Logistic Growth Model)
- Montroll (Generalized Growth Model)
- Physics-Informed Neural Networks (PINNs)

## 🎯 Objective
- Model tumor growth using differential equations
- Fit models to real-world biological data
- Compare accuracy between models
- Understand growth saturation behavior

## 📊 Dataset
- Source: Marušić et al. (1994)
- 45 observations of tumor volume over 60 days
- Format: Time vs Tumor Volume

## 🧠 Models Implemented

### 1. Verhulst Model
dp/dt = k p (1 - p/C)

### 2. Montroll Model
dp/dt = k p (1 - (p/C)^θ)

### 3. PINN Model
- Combines neural networks + physics constraints
- Uses residual-based loss function

## ⚙️ Methodology
1. Load dataset
2. Fit Verhulst model
3. Fit Montroll model
4. Compute error metrics
5. Compare results
6. Visualize growth curves

## 📈 Results

- Montroll model provides better fit than Verhulst
- Captures inflection point more accurately
- PINN approach enables learning parameters from data

## 📉 Example Output

- Raw tumor growth curve
- Model comparison graph
- Error metrics

## 🛠️ Installation

```bash
pip install -r requirements.txt
