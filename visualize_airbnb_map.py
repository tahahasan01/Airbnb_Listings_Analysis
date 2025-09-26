import pandas as pd
import folium
from folium.plugins import MarkerCluster

# Load the data
df = pd.read_csv("airbnb_listings_pakistan.csv")

# Parse price to numeric value (remove $ and commas)
df["price_num"] = df["price"].replace('[\$,]', '', regex=True).astype(float)

# Center map
center_lat = df["latitude"].mean()
center_lon = df["longitude"].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=7)

marker_cluster = MarkerCluster().add_to(m)

# Helper: choose marker color by price
def get_color(price):
    if price < 30:
        return "green"
    elif price < 80:
        return "orange"
    else:
        return "red"

# Add markers
for _, row in df.iterrows():
    if pd.notnull(row["latitude"]) and pd.notnull(row["longitude"]):
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=folium.Popup(
                f"<b>{row['name']}</b><br>"
                f"{row['city']}<br>"
                f"{row['price']}<br>"
                f"<a href='{row['url']}' target='_blank'>View Listing</a>",
                max_width=300,
            ),
            icon=folium.Icon(color=get_color(row["price_num"]), icon="home", prefix="fa"),
        ).add_to(marker_cluster)

m.save("airbnb_pakistan_map_colored.html")
print("✅ Saved color-coded map to airbnb_pakistan_map_colored.html")
