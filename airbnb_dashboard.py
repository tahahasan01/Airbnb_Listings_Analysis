import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Page Setup ---
st.set_page_config(
    page_title="Pakistan Airbnb Analysis",
    page_icon="📊",
    layout="wide",
)

# --- Data Loading ---
@st.cache_data
def load_data():
    df_combined_cleaned = pd.read_csv('df_combined_cleaned.csv')
    city_analysis_major = pd.read_csv('city_analysis_major.csv')
    return df_combined_cleaned, city_analysis_major

df_combined_cleaned, city_analysis_major = load_data()


# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Price Analysis", "Geographic Analysis", "Model Predictions", "Key Insights"])


# --- Overview Page ---
if page == "Overview":
    st.title("📊 Overview")

    # Key Metrics
    st.header("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Listings", f"{len(df_combined_cleaned):,}")
    col2.metric("Average Price", f"${df_combined_cleaned['price'].mean():.2f}")
    col3.metric("Cities Covered", f"{df_combined_cleaned['city'].nunique()}")
    col4.metric("Average Rating", f"{df_combined_cleaned['stars'].mean():.2f}")

    # Dataset Preview
    st.header("Dataset Preview")
    st.dataframe(df_combined_cleaned.head())

    # Price Statistics and Room Type Distribution
    st.header("Price and Room Type Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Price Statistics")
        st.write(df_combined_cleaned['price'].describe())

    with col2:
        st.subheader("Room Type Distribution")
        fig = px.pie(df_combined_cleaned, names='roomType', title='Distribution of Room Types')
        st.plotly_chart(fig)


# --- Price Analysis Page ---
elif page == "Price Analysis":
    st.title("💰 Price Analysis")

    # Price Distribution Histogram
    st.header("Price Distribution")
    fig = px.histogram(df_combined_cleaned, x='price', nbins=50, title='Distribution of Prices')
    st.plotly_chart(fig)

    # Price Comparison by Room Type
    st.header("Price Comparison by Room Type")
    fig = px.box(df_combined_cleaned, x='roomType', y='price', title='Price Distribution by Room Type')
    st.plotly_chart(fig)

    # Average Price by City
    st.header("Average Price by City")
    avg_price_by_city = df_combined_cleaned.groupby('city')['price'].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(avg_price_by_city, x='city', y='price', title='Average Price by City')
    st.plotly_chart(fig)

    # Price vs. Rating Correlation
    st.header("Price vs. Rating")
    fig = px.scatter(df_combined_cleaned, x='stars', y='price', title='Price vs. Rating')
    st.plotly_chart(fig)


# --- Geographic Analysis Page ---
elif page == "Geographic Analysis":
    st.title("🗺️ Geographic Analysis")

    # Interactive Map
    st.header("Interactive Map of Listings")
    map_center = [df_combined_cleaned['location/lat'].mean(), df_combined_cleaned['location/lng'].mean()]
    airbnb_map = folium.Map(location=map_center, zoom_start=6)

    # Price Ranges for Color Coding
    def get_color(price):
        if price > 75:
            return 'red'
        elif price > 50:
            return 'orange'
        elif price > 25:
            return 'green'
        else:
            return 'blue'

    for idx, row in df_combined_cleaned.iterrows():
        folium.CircleMarker(
            location=[row['location/lat'], row['location/lng']],
            radius=5,
            color=get_color(row['price']),
            fill=True,
            fill_color=get_color(row['price']),
            popup=f"<b>{row['name']}</b><br>Price: ${row['price']}<br>Room Type: {row['roomType']}"
        ).add_to(airbnb_map)

    folium_static(airbnb_map)

    # Listings Count by City
    st.header("Listings Count by City")
    listings_by_city = df_combined_cleaned['city'].value_counts().reset_index()
    listings_by_city.columns = ['city', 'count']
    fig = px.bar(listings_by_city, x='city', y='count', title='Number of Listings by City')
    st.plotly_chart(fig)

    # Average Price Comparison Across Cities
    st.header("Average Price Comparison")
    avg_price_by_city = df_combined_cleaned.groupby('city')['price'].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(avg_price_by_city, x='city', y='price', title='Average Price by City')
    st.plotly_chart(fig)


# --- Model Predictions Page ---
elif page == "Model Predictions":
    st.title("🤖 Model Predictions")

    # Model Training
    st.header("Model Training")
    with st.spinner("Training the model..."):
        features = ['roomType', 'stars', 'city', 'isHostedBySuperhost', 'location/lat', 'location/lng', 'numberOfGuests']
        target = 'price'
        X = df_combined_cleaned[features]
        y = df_combined_cleaned[target]

        categorical_features = ['roomType', 'city', 'isHostedBySuperhost']
        numerical_features = ['stars', 'location/lat', 'location/lng', 'numberOfGuests']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', SimpleImputer(strategy='median'), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        model = LinearRegression()
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

    st.success("Model training complete!")

    # Model Performance
    st.header("Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Absolute Error (MAE)", f"${mae:.2f}")
    col2.metric("Mean Squared Error (MSE)", f"${mse:.2f}")
    col3.metric("R-squared (R²)", f"{r2:.2f}")

    # Predictions vs. Actuals
    st.header("Predictions vs. Actuals")
    fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Prices', 'y': 'Predicted Prices'}, title='Actual vs. Predicted Prices')
    fig.add_shape(type='line', x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(color='red', dash='dash'))
    st.plotly_chart(fig)

    # Interactive Price Prediction
    st.header("Interactive Price Prediction")
    room_type = st.selectbox("Room Type", df_combined_cleaned['roomType'].unique())
    stars = st.slider("Rating", 1.0, 5.0, 4.5, 0.1)
    city = st.selectbox("City", df_combined_cleaned['city'].unique())
    superhost = st.selectbox("Superhost", ["Yes", "No"])
    lat = st.number_input("Latitude", value=30.0)
    lng = st.number_input("Longitude", value=70.0)
    guests = st.slider("Number of Guests", 1, 16, 2)

    if st.button("Predict Price"):
        input_data = pd.DataFrame({
            'roomType': [room_type],
            'stars': [stars],
            'city': [city],
            'isHostedBySuperhost': [superhost],
            'location/lat': [lat],
            'location/lng': [lng],
            'numberOfGuests': [guests]
        })
        prediction = pipeline.predict(input_data)[0]
        st.success(f"Predicted Price: ${prediction:.2f}")


# --- Key Insights Page ---
elif page == "Key Insights":
    st.title("💡 Key Insights")

    st.header("Business Recommendations")
    st.write("""
    - **Focus on High-Demand Cities**: Islamabad and Lahore show the highest number of listings and reviews, indicating strong demand.
    - **"Entire Place" Listings are Most Profitable**: These listings have the highest average price and are likely to attract families or groups.
    - **Superhost Status Matters**: While not explicitly analyzed here, being a Superhost can build trust and attract more bookings.
    - **Competitive Pricing is Key**: With a high number of listings in major cities, competitive pricing is crucial to attract guests.
    """)

    st.header("Market Analysis Summary")
    st.write("""
    The Airbnb market in Pakistan is concentrated in a few major cities, with Lahore, Islamabad, and Karachi being the most prominent. Prices vary significantly by city and room type, with "Entire place" listings commanding the highest prices. The market is competitive, so hosts need to focus on providing value and excellent service to stand out.
    """)

    st.header("High-Value Market Identification")
    st.write("""
    **Islamabad** stands out as a high-value market. It has a high number of listings, strong demand (as indicated by the number of reviews), and a high average price. This makes it an attractive city for new Airbnb hosts.
    """)