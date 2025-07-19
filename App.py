# Top of App.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

# Train model
df = pd.read_csv("Life Expectancy Data.csv")
df.columns = df.columns.str.strip()
df = df[df["Life expectancy"].notna()]
df = df.fillna(df.mean(numeric_only=True))

features = ['Alcohol', 'BMI', 'GDP', 'Schooling']
X = df[features]
y = df['Life expectancy']
model = DecisionTreeRegressor()
model.fit(X, y)

# Streamlit UI (your existing code with sliders)
# ...


st.title("🌍 Life Expectancy Predictor")

st.markdown("### 📋 Enter your health and economic details below:")

alcohol = st.slider(
    "🍷 Alcohol Consumption (litres per person/year)",
    0.0, 20.0, 5.0,
    help="Average pure alcohol consumption per person annually"
)

bmi = st.slider(
    "⚖️ Body Mass Index (BMI)",
    10.0, 50.0, 22.0,
    help="18.5–24.9 is considered healthy"
)

gdp = st.slider(
    "💰 GDP per Capita (in US dollars)",
    0.0, 150000.0, 10000.0,
    help="Income per person, used as economic strength indicator"
)

schooling = st.slider(
    "🎓 Years of Schooling",
    0.0, 20.0, 12.0,
    help="Total number of years of formal education (12 = high school)"
)
if st.button("Predict Life Expectancy"):
    input_data = pd.DataFrame([[alcohol, bmi, gdp, schooling]], columns=features)
    prediction = model.predict(input_data)[0]

    st.markdown("### 🧠 Predicted Life Expectancy")
    st.success(f"🎯 **{prediction:.2f} years**")

    # Optional: progress bar to visualize
    st.progress(min(int(prediction), 100))  # cap at 100
