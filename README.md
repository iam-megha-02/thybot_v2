# ThyBot: Thyroid Health Analysis & Diet Assistant

ThyBot is a personalized, LLM-powered health assistant built with Streamlit that helps users understand their thyroid condition, analyze meals for thyroid friendliness, and receive intelligent diet recommendations.

## 📌 Table of Contents

- [1. Background](#-background)
- [2. Problem Statement](#-problem-statement)
- [3. Objectives](#-objectives)
- [4. Features](#-features)
- [5. Tech Stack](#-tech-stack)
- [6. Project Structure](#-project-structure)
- [7. Getting Started](#-getting-started)
- [8. Example Data](#-example-data)
- [9. Future Enhancements](#-future-enhancements)

## 🔍 Background

Thyroid disorders affect over **42 million** Indians and are often underdiagnosed. Managing thyroid-related conditions like **hypothyroidism** and **hyperthyroidism** requires dietary care, consistent monitoring, and accessible education — areas where many patients struggle due to lack of guidance and personalized support.

---

## ❓ Problem Statement

There is a lack of **interactive**, **AI-powered**, and **diet-aware** tools tailored for thyroid patients. Existing platforms rarely provide dietary recommendations based on **thyroid type**, and none incorporate intelligent document/question understanding.

---

## 🎯 Objectives

- Predict thyroid condition using a medical profile
- Classify Indian meals as thyroid-friendly or not
- Recommend foods to add/remove from diet
- Allow document uploads for analysis using LLM
- Provide a clean and intuitive interface for regular use

---

## ✨ Features

✔️ **Thyroid Condition Predictor**  
✔️ **Meal Classifier with Nutrient Insights**  
✔️ **LLM Support** (via Groq API) for unknown items  
✔️ **Document Upload + QA Chat** for patient reports  
✔️ **Dynamic Meal List with Add/Remove Options**  
✔️ **Personalized Suggestions Based on Thyroid Type**  
✔️ **Example Datasets Included** for quick testing

---

## ⚙️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: Groq (LLaMA3)
- **Data Processing**: Pandas
- **Vector Search**: FAISS
- **Others**: LangChain, dotenv, tiktoken

