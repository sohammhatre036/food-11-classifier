# 🍔 Food-11 Classifier (PyTorch + Streamlit)

This project trains a CNN (ResNet18 / VGG-lite) on the **Food-11 dataset** and serves a web app with **Streamlit** for real-time food image classification.

## Files
- `food_classification.py` → Training script  
- `streamlit_full.py` → Streamlit app (predict + metrics)  
- `best_model_res.h5` → Saved model (ignored in repo by default)  

## How to run
```bash
pip install -r requirements.txt
python food_classification.py --data_dir ./food-11
streamlit run streamlit_full.py
