# =========================
# FINAL DATA CLEANING + EDA
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# -------------------------
# Load Dataset
# -------------------------

df = pd.read_csv("mergedddd.csv")   

# -------------------------
# Keep only valid rows (0–2000 and 2108–2602), remove empty block
# -------------------------
df = pd.concat([df.loc[:2000], df.loc[2108:]], axis=0).reset_index(drop=True)

# -------------------------
# Drop duplicates
# -------------------------
df.drop_duplicates(inplace=True)

# -------------------------
# Drop rows missing critical values
# -------------------------
df.dropna(subset=["Price", "Area Size", "Property Type", "Location"], inplace=True)

# -------------------------
# Clean Property Type
# -------------------------

def clean_property_type(pt):
    if pd.isnull(pt):
        return np.nan
    pt = str(pt).strip().lower()
    if "house" in pt:
        return "House"
    elif "flat" in pt or "apartment" in pt:
        return "Flat"
    elif "plot" in pt:
        return "Plot"
    elif "commercial" in pt:
        return "Commercial"
    else:
        return pt.title()

df["Property Type"] = df["Property Type"].apply(clean_property_type)

# -------------------------
# Clean Price
# -------------------------

def convert_price(price_str):
    if pd.isnull(price_str):
        return np.nan
    price_str = price_str.replace("PKR", "").strip()
    if "Crore" in price_str:
        num = float(re.findall(r"[\d.]+", price_str)[0])
        return num * 1e7
    elif "Lakh" in price_str:
        num = float(re.findall(r"[\d.]+", price_str)[0])
        return num * 1e5
    elif "Thousand" in price_str:
        num = float(re.findall(r"[\d.]+", price_str)[0])
        return num * 1e3
    else:
        try:
            return float(re.findall(r"[\d.]+", price_str)[0])
        except:
            return np.nan

df["Price_numeric"] = df["Price"].apply(convert_price)

# -------------------------
# Clean Area Size
# -------------------------
def convert_area(area_str):
    if pd.isnull(area_str):
        return np.nan
    tokens = area_str.split()
    try:
        value = float(tokens[0])
    except:
        return np.nan
    
    if "Marla" in area_str:
        return value * 272.25
    elif "Kanal" in area_str:
        return value * 5445
    elif "Sqft" in area_str or "Square Feet" in area_str:
        return value
    else:
        return np.nan

df["Area_sqft"] = df["Area Size"].apply(convert_area)

# -------------------------
# Convert numeric cols
# -------------------------

for col in ["Bedrooms", "Bathrooms", "Parking Spaces", "Floors"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# -------------------------
# Keep only useful columns
# -------------------------

useful_cols = [
    "Card Title", "Property Type", "Location", "Price_numeric", "Area_sqft",
    "Bedrooms", "Bathrooms", "Parking Spaces", "Floors", "Year",
    # Some important amenities
    "Furnished", "Servant Quarters", "Dining Room", "Kitchens",
    "Study Room", "Prayer Room", "Gym", "Store Rooms",
    "Lounge or Sitting Room", "Laundry Room", "Community Lawn or Garden",
    "Community Gym", "Nearby Schools", "Nearby Hospitals",
    "Nearby Shopping Malls", "Nearby Restaurants", "Distance From Airport (kms)",
    "Security Staff", "Kids Play Area", "Community Swimming Pool", "Mosque"
]

df = df[useful_cols]

# -------------------------
# Save as Final_Zameen_Data.csv
# -------------------------
df.to_csv("Final_Zameen_Data.csv", index=False)
print("✅ Cleaned dataset saved as Final_Zameen_Data.csv")
print("Rows & Cols:", df.shape)

# ==========================================================
# 📊 EDA
# ==========================================================

# --- Univariate Analysis ---
plt.figure(figsize=(8,5))
sns.histplot(df["Price_numeric"].dropna()/1e7, bins=50, kde=True)
plt.title("Distribution of Property Prices (in Crores PKR)")
plt.xlabel("Price (Crores)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df["Area_sqft"].dropna(), bins=50, kde=True)
plt.title("Distribution of Property Area (sqft)")
plt.xlabel("Area (sqft)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="Property Type", data=df)
plt.title("Property Type Distribution")
plt.xticks(rotation=45)
plt.show()

# --- Bivariate Analysis ---
plt.figure(figsize=(8,6))
sns.scatterplot(x="Area_sqft", y="Price_numeric", hue="Property Type", data=df, alpha=0.6)
plt.title("Price vs Area Size")
plt.xlabel("Area (sqft)")
plt.ylabel("Price (PKR)")
plt.ylim(0, df["Price_numeric"].quantile(0.95))  # remove extreme outliers
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(x="Bedrooms", y="Price_numeric", data=df)
plt.title("Price vs Bedrooms")
plt.xlabel("Bedrooms")
plt.ylabel("Price (PKR)")
plt.ylim(0, df["Price_numeric"].quantile(0.95))
plt.show()

top_locations = df["Location"].value_counts().head(10).index
plt.figure(figsize=(12,6))
sns.boxplot(x="Location", y="Price_numeric", data=df[df["Location"].isin(top_locations)])
plt.title("Price Distribution by Top Locations")
plt.xticks(rotation=45)
plt.ylim(0, df["Price_numeric"].quantile(0.95))
plt.show()

# --- Multivariate Analysis ---
numeric_cols = ["Price_numeric","Area_sqft","Bedrooms","Bathrooms","Parking Spaces"]
plt.figure(figsize=(8,6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# --- Feature Impact Example ---
feature_cols = ["Gym","Mosque","Community Lawn or Garden","Community Swimming Pool","Security Staff"]
for col in feature_cols:
    if col in df.columns:
        avg_prices = df.groupby(col)["Price_numeric"].mean()
        plt.figure(figsize=(5,3))
        avg_prices.plot(kind="bar", color=["red","green"])
        plt.title(f"Average Price by {col}")
        plt.ylabel("Price (PKR)")
        plt.show()
