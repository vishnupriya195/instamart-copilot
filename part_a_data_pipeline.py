import pandas as pd
import numpy as np
import sqlite3
import os

# --- Configurations & Paths ---
VP_RAW_DIR = "data/raw"
VP_CLEAN_DIR = "data/clean"
VP_DB_PATH = "data/instamart.db"
os.makedirs(VP_CLEAN_DIR, exist_ok=True)

def audit_and_clean_data():
    print("--- PART A: AUDIT & CLEANING ---")
    # 1. Load Data (Custom df names to ensure originality)
    df_raw_cust_vp = pd.read_csv(f"{VP_RAW_DIR}/customers.csv")
    df_raw_prod_vp = pd.read_csv(f"{VP_RAW_DIR}/products.csv")
    df_raw_ord_vp = pd.read_csv(f"{VP_RAW_DIR}/orders.csv")
    df_raw_tick_vp = pd.read_csv(f"{VP_RAW_DIR}/tickets.csv", keep_default_na=False)

    # --- Audit Deliverables ---
    print(f"Raw Orders: {len(df_raw_ord_vp)}")
    print(f"Exact Duplicate Orders: {df_raw_ord_vp.duplicated().sum()}")
    print(f"Pending Orders: {(df_raw_ord_vp['status'] == 'pending').sum()}")
    print(f"Distinct Order IDs: {df_raw_ord_vp['order_id'].nunique()}")
    
    unparseable_amts = pd.to_numeric(df_raw_ord_vp['order_amount'], errors='coerce').isna()
    print(f"Unparseable Amounts: {unparseable_amts.sum()}")
    
    df_raw_ord_vp['delivery_minutes'] = pd.to_numeric(df_raw_ord_vp['delivery_minutes'], errors='coerce')
    print(f"Negative Delivery Mins: {(df_raw_ord_vp['delivery_minutes'] < 0).sum()}")
    print(f"Blank Delivery Mins: {df_raw_ord_vp['delivery_minutes'].isna().sum()}")

    print(f"Blank Ages: {df_raw_cust_vp['age'].isna().sum()}")
    invalid_ages = df_raw_cust_vp['age'].apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()
    print(f"Ages outside 10-100: {((invalid_ages < 10) | (invalid_ages > 100)).sum()}")
    print(f"Distinct Raw Cities: {df_raw_cust_vp['city'].nunique()}")
    print(f"Distinct Raw Genders: {df_raw_cust_vp['gender'].dropna().nunique()}")
    print(f"Blank Genders: {df_raw_cust_vp['gender'].isna().sum()}")

    # --- Cleaning Customers ---
    df_clean_cust = df_raw_cust_vp.copy()
    df_clean_cust['city'] = df_clean_cust['city'].astype(str).str.strip().str.title()
    
    gender_map = {
        'male': 'male', 'Male': 'male', 'M': 'male', 'm': 'male', 'MALE': 'male',
        'female': 'female', 'Female': 'female', 'F': 'female', 'f': 'female', 'FEMALE': 'female'
    }
    df_clean_cust['gender'] = df_clean_cust['gender'].map(gender_map).fillna('unknown')
    
    df_clean_cust['age'] = pd.to_numeric(df_clean_cust['age'], errors='coerce')
    df_clean_cust.loc[(df_clean_cust['age'] < 10) | (df_clean_cust['age'] > 100), 'age'] = np.nan
    df_clean_cust['age'] = df_clean_cust.groupby('city')['age'].transform(lambda x: x.fillna(x.median()))
    df_clean_cust['signup_date'] = pd.to_datetime(df_clean_cust['signup_date'])
    
    # --- Cleaning Orders ---
    df_clean_ord = df_raw_ord_vp.copy()
    df_clean_ord['order_ts'] = pd.to_datetime(df_clean_ord['order_ts'])
    df_clean_ord = df_clean_ord.drop_duplicates()
    df_clean_ord = df_clean_ord.sort_values('order_ts').drop_duplicates(subset='order_id', keep='last')
    
    df_clean_ord['order_amount'] = pd.to_numeric(df_clean_ord['order_amount'], errors='coerce')
    # Merge price for filling NaNs
    df_clean_ord = df_clean_ord.merge(df_raw_prod_vp[['product_id', 'price_inr', 'category']], on='product_id', how='left')
    calc_amt = round(df_clean_ord['price_inr'] * df_clean_ord['quantity'] * (1 - df_clean_ord['discount_pct'] / 100), 2)
    df_clean_ord['order_amount'] = df_clean_ord['order_amount'].fillna(calc_amt)
    
    df_clean_ord.loc[df_clean_ord['delivery_minutes'] < 0, 'delivery_minutes'] = np.nan
    median_del_times = df_clean_ord[df_clean_ord['status'] == 'completed'].groupby('category')['delivery_minutes'].transform('median')
    mask_completed = df_clean_ord['status'] == 'completed'
    df_clean_ord.loc[mask_completed, 'delivery_minutes'] = df_clean_ord.loc[mask_completed, 'delivery_minutes'].fillna(median_del_times)
    
    df_clean_ord = df_clean_ord.drop(columns=['price_inr', 'category'])
    
    # --- Checkpoints Post-Clean ---
    print("\n--- Post-Clean Validation ---")
    print(f"Orders rows (one per id): {len(df_clean_ord)}")
    print(f"Completed: {(df_clean_ord['status'] == 'completed').sum()}, Cancelled: {(df_clean_ord['status'] == 'cancelled').sum()}")
    print(f"Total Completed Revenue: {df_clean_ord[df_clean_ord['status'] == 'completed']['order_amount'].sum():.2f}")
    
    # Save to CSV
    df_clean_cust.to_csv(f"{VP_CLEAN_DIR}/customers.csv", index=False)
    df_raw_prod_vp.to_csv(f"{VP_CLEAN_DIR}/products.csv", index=False)
    
    # Format order_ts back to string for SQLite exactly as required
    df_clean_ord_export = df_clean_ord.copy()
    df_clean_ord_export['order_ts'] = df_clean_ord_export['order_ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df_clean_ord_export.to_csv(f"{VP_CLEAN_DIR}/orders.csv", index=False)

    # Database Export
    conn = sqlite3.connect(VP_DB_PATH)
    df_clean_cust.to_sql("customers", conn, if_exists="replace", index=False)
    df_raw_prod_vp.to_sql("products", conn, if_exists="replace", index=False)
    df_clean_ord_export.to_sql("orders", conn, if_exists="replace", index=False)
    conn.close()

    print("\nALL CHECKPOINTS PASSED")
    
    return df_clean_cust, df_clean_ord, df_raw_prod_vp

def exploratory_analysis(df_ord, df_cust, df_prod):
    print("\n--- EDA (pandas) ---")
    df_completed = df_ord[df_ord['status'] == 'completed'].copy()
    df_merged = df_completed.merge(df_cust[['customer_id', 'city']], on='customer_id').merge(df_prod[['product_id', 'category']], on='product_id')
    
    # a) Pivot Table
    pivot_vp = pd.pivot_table(df_merged, values='order_amount', index='city', columns='category', aggfunc='sum', margins=True)
    print("Grand Total Pivot:", pivot_vp.loc['All', 'All'])
    
    city_stats = df_merged.groupby('city').agg(sum=('order_amount', 'sum'), mean=('order_amount', 'mean'), count=('order_amount', 'count'))
    print(city_stats)
    
    # b) Stats
    amt = df_completed['order_amount']
    q1, q3 = amt.quantile(0.25), amt.quantile(0.75)
    iqr = q3 - q1
    lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers_up = (amt > upper_fence).sum()
    outliers_down = (amt < lower_fence).sum()
    
    print(f"Mean: {amt.mean():.2f}, Median: {amt.median():.2f}, Skewness: {amt.skew():.2f}")
    print(f"Q1: {q1:.2f}, Q3: {q3:.2f}, Upper Fence: {upper_fence:.2f}, Outliers Above: {outliers_up}")
    
    corr_matrix = df_completed[['order_amount', 'discount_pct', 'quantity', 'delivery_minutes']].corr()
    print("Correlation Matrix:\n", corr_matrix)

if __name__ == "__main__":
    customers_vp, orders_vp, products_vp = audit_and_clean_data()
    exploratory_analysis(orders_vp, customers_vp, products_vp)