import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder

# Analysis Section
data = pd.read_csv("E:\DATASETS\CAR_PRICE.csv")

data[["carCompany", "carModel"]] = data["CarName"].str.split(" ", 1, expand=True)
data = data.drop(["CarName"], axis=1)

X = data.drop("price", axis=1)
y = data["price"]

categorical_columns = [
    "fueltype",
    "aspiration",
    "carbody",
    "drivewheel",
    "enginelocation",
    "enginetype",
    "fuelsystem",
]
numeric_columns = [
    "symboling",
    "doornumber",
    "wheelbase",
    "carlength",
    "carwidth",
    "carh8",
    "curbw8",
    "cylindernumber",
    "enginesize",
    "boreratio",
    "stroke",
    "compressionratio",
    "horsepower",
    "peakrpm",
    "citympg",
    "highwaympg",
]

encoder = OneHotEncoder(drop="first", sparse=False)
X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_columns]))
X_encoded.columns = encoder.get_feature_names_out(categorical_columns)
X_numeric = X[numeric_columns]
X_processed = pd.concat([X_numeric, X_encoded], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

rf_regressor = RandomForestRegressor(
    n_estimators=150,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
)

rf_regressor.fit(X_train, y_train)

y_pred = rf_regressor.predict(X_test)

# Streamlit app Section
st.title("Car Price Prediction App")
image = "RCAR_IMG.jpg"
st.image(image, caption="Car Image", use_column_width=True)

st.write("Welcome to the Car Price Prediction App!")
st.write(
    "This app uses a trained machine learning model to predict the price of a car based on the provided features. "
    "Please enter the details of the car and the predicted price will be displayed."
)

# Graph Section
for i in range(0, len(numeric_columns), 2):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for j, feature in enumerate(numeric_columns[i : i + 2]):
        axes[j].scatter(data["price"], data[feature], color="blue")
        axes[j].set_xlabel("Car Price ($)")
        axes[j].set_ylabel(feature)
        axes[j].set_title(f"{feature} vs Car Price")
    st.pyplot(fig)

st.markdown(
    "<h3 style='text-align: center;'>R-squared Score: {:.3f}</h2>".format(
        r2_score(y_test, y_pred)
    ),
    unsafe_allow_html=True,
)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color="red", label="Predicted")
plt.plot(y_test, y_test, color="blue", linewidth=2, label="Actual")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.title("Actual vs Predicted Car Prices")
plt.legend()
st.pyplot(plt)

# Prediction Section
st.markdown(
    "<h3 style='text-align: center;'>Enter Car Details:</h3>", unsafe_allow_html=True
)
input_features = {}
col1, col2 = st.columns(2)
for feature in numeric_columns:
    input_features[feature] = col1.number_input(f"Enter {feature}", value=0)

for feature in categorical_columns:
    input_features[feature] = col2.selectbox(
        f"Select {feature}", data[feature].unique()
    )

input_df = pd.DataFrame(input_features, index=[0])

st.write("User Input:")
st.write(input_df)

input_encoded = pd.DataFrame(encoder.transform(input_df[categorical_columns]))
input_encoded.columns = encoder.get_feature_names_out(categorical_columns)
input_numeric = input_df[numeric_columns]
input_processed = pd.concat([input_numeric, input_encoded], axis=1)

predicted_price = rf_regressor.predict(input_processed)

st.markdown(
    "<h3 style='text-align: center;'>Predicted Car Price:</h3>", unsafe_allow_html=True
)
st.write(
    f"<h1 style='text-align: center; font-size: 36px; color: red;'>${predicted_price[0]:,.2f}</h1>",
    unsafe_allow_html=True,
)
