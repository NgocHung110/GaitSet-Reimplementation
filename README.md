# 🧍 GaitSet Reimplementation on CASIA-B

This project reimplements the [GaitSet](https://arxiv.org/abs/1811.06186) model for gait recognition using the CASIA-B dataset. It includes reproduction of the original results, experiments with new gait data, and fine-tuning for improved generalization.

---

## 🎯 Project Objectives

- ✅ Reproduce the GaitSet model using the official code and CASIA-B dataset.
- ✅ Modify and retrain the model to verify results from the original paper.
- ✅ Test model generalization using custom-created gait silhouette data.
- ✅ Fine-tune the model if performance on new data is suboptimal.

---

## 📚 What is GaitSet?

GaitSet is a deep learning framework for gait recognition that treats gait as an unordered set of silhouette images, instead of sequential frames. This allows the model to extract view-invariant features and achieve state-of-the-art performance on multi-view datasets like CASIA-B.

Key features:
- Uses Set Pooling to aggregate information from multiple silhouette images.
- Handles cross-view gait recognition without requiring frame alignment.

---

## 🔧 What I Did

| Step | Description |
|------|-------------|
| 1️⃣ | Cloned the official GaitSet GitHub repository |
| 2️⃣ | Modified training and configuration files to suit local environment |
| 3️⃣ | Re-trained the model on CASIA-B dataset |
| 4️⃣ | Collected and processed custom gait silhouette data for generalization testing |
| 5️⃣ | Evaluated performance of pre-trained model on new data |
| 6️⃣ | Fine-tuned the model to improve accuracy on custom dataset |

---

## 📊 Experimental Results

| Experiment               | Accuracy (%) | Notes                                  |
|--------------------------|--------------|----------------------------------------|
| CASIA-B Reproduction     | 95.1%        | For NM type                            |
| Custom Gait Data (raw)   | 60.42%       | Mean Rank-1 (all condition)            |
| Fine-tuned on new data   | 88.89%       | Performance improved significantly     |

## 🧪 Dataset

- **CASIA-B**: Used for model reproduction and evaluation.  
- **Custom Gait Data**: Created using synthetic silhouette sequences for generalization testing.  

---

## 🔗 Google Colab Link

If available, add your working Colab version here:

[👉 Google Colab Notebook (view only)](https://colab.research.google.com/drive/1i01fu-jPtQ95iRh9UEOU56r6xHK8NbYh?authuser=1#scrollTo=dKK9QJHKiBys)

---

## 👨‍💻 Author

**Nguyen Ngoc Hung**  
Final-year Computer Engineering Student  
University of Science and Technology – The University of Danang  
💡 Interests: AI, Machine Learning, Computer Vision, Embedded Systems

📧 nguyenngochung110@gmail.com

---

## 📌 Acknowledgments

- [Original GaitSet GitHub repo](https://github.com/AbnerHqC/GaitSet)
- [CASIA-B Dataset Info](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp)