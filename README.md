# Predictive Modeling of Consumer Impulse Buying Behavior in E-Commerce

This repository contains a complete Business Analytics case study that predicts consumer impulse buying behavior. By analyzing demographic data, psychological states (e.g., mood), and promotional triggers (e.g., flash sales), this project builds a classification model to help e-commerce retailers optimize their promotional strategies and maximize Average Order Value (AOV).

## 📊 Business Problem
Retailers heavily rely on impulse buying to increase revenue, yet frequently waste marketing budgets on generic, blanket promotions. The objective of this project is to shift from mass marketing to data-driven, targeted interventions by predicting exactly *when* and *why* a customer is most likely to make an unplanned purchase. 

## 📁 Dataset Information
The project utilizes a dataset of 10,000 records (`impulse_buying_dataset_10000.csv`). 

**Features:**
*   **Demographics:** `Age`, `Gender`, `Employment_Status`, `Disposable_Income`
*   **Behavioral:** `Preferred_Platform`, `Shopping_Frequency_Per_Month`
*   **Psychological & Triggers:** `Current_Mood`, `Primary_Trigger`, `Product_Category`
*   **Target Variable:** `Made_Impulse_Purchase` (Yes/No)

## 🛠️ Project Workflow

1.  **Data Collection / Generation:** Synthetic dataset generation capturing underlying statistical correlations between consumer mood, promotional triggers, and purchase probability.
2.  **Data Preprocessing:** Handling categorical variables using One-Hot Encoding (`pd.get_dummies`) and scaling numerical features.
3.  **Exploratory Data Analysis (EDA):** Visualizing purchase distributions across different moods, income brackets, and shopping platforms using `seaborn` and `matplotlib`.
4.  **Predictive Modeling:** Training and evaluating classification models (`scikit-learn`) such as Logistic Regression, Decision Trees, or Random Forests to predict the `Made_Impulse_Purchase` target variable.
5.  **Business Insights Generation:** Translating feature importance (e.g., the high conversion rate of "Limited Time Flash Sales" among "Bored" users) into actionable UI/UX and marketing recommendations.

## 🚀 How to Run the Project

### Prerequisites
Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

```bash
# Clone the repository
git clone [https://github.com/yourusername/impulse-buying-prediction.git](https://github.com/yourusername/impulse-buying-prediction.git)

# Navigate into the directory
cd impulse-buying-prediction

# Install required dependencies
pip install pandas numpy matplotlib seaborn scikit-learn
