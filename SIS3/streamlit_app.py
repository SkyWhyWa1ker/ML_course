import streamlit as st
import requests

st.set_page_config(page_title="Iris Classifier", page_icon="🌸")

st.title("🌸 Iris Flower Classifier")
st.markdown("Enter the flower measurements below to predict the species.")

# Quick example buttons
st.markdown("**Quick examples:**")
col1, col2, col3 = st.columns(3)
if col1.button("🌺 Setosa example"):
    st.session_state.sl, st.session_state.sw = 5.1, 3.5
    st.session_state.pl, st.session_state.pw = 1.4, 0.2
if col2.button("🌼 Versicolor example"):
    st.session_state.sl, st.session_state.sw = 6.0, 2.9
    st.session_state.pl, st.session_state.pw = 4.5, 1.5
if col3.button("🌸 Virginica example"):
    st.session_state.sl, st.session_state.sw = 6.5, 3.0
    st.session_state.pl, st.session_state.pw = 5.5, 2.0

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=st.session_state.get("sl", 5.1), step=0.1)
    sepal_width  = st.number_input("Sepal Width (cm)",  min_value=0.0, max_value=10.0, value=st.session_state.get("sw", 3.5), step=0.1)
with col2:
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=st.session_state.get("pl", 1.4), step=0.1)
    petal_width  = st.number_input("Petal Width (cm)",  min_value=0.0, max_value=10.0, value=st.session_state.get("pw", 0.2), step=0.1)

if st.button("Predict", use_container_width=True, type="primary"):
    payload = {
        "sepal_length": sepal_length,
        "sepal_width":  sepal_width,
        "petal_length": petal_length,
        "petal_width":  petal_width,
    }
    try:
        response = requests.post("http://fastapi:8000/predict", json=payload, timeout=5)
        result = response.json()
        class_name = result.get("class_name", "unknown")
        prediction = result.get("prediction")

        emoji_map = {"setosa": "🌺", "versicolor": "🌼", "virginica": "🌸"}
        emoji = emoji_map.get(class_name, "🌿")

        st.success(f"{emoji} Predicted species: **{class_name.capitalize()}** (class {prediction})")
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
