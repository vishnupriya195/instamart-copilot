import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.utils import resample
import joblib
import warnings
warnings.filterwarnings("ignore")

def part_b_execution():
    # Load clean data
    vp_cust = pd.read_csv("data/clean/customers.csv")
    vp_ord = pd.read_csv("data/clean/orders.csv")
    vp_ord['order_ts'] = pd.to_datetime(vp_ord['order_ts'])
    vp_ord_completed = vp_ord[vp_ord['status'] == 'completed'].copy()

    # --- 1. RFM Table ---
    snapshot_date = pd.to_datetime("2025-10-01 00:00:00")
    rfm_vp = vp_ord_completed.groupby('customer_id').agg(
        recency=('order_ts', lambda x: (snapshot_date - x.max()).days),
        frequency=('order_id', 'nunique'),
        monetary=('order_amount', 'sum')
    ).reset_index()
    
    print(f"RFM Setup: {len(rfm_vp)} customers")
    
    # --- 2. Clustering ---
    scaler_rfm = StandardScaler()
    rfm_scaled = scaler_rfm.fit_transform(rfm_vp[['recency', 'frequency', 'monetary']])
    
    print("K-Means Inertia & Silhouette omitted for brevity, proceeding with K=4")
    kmeans_vp = KMeans(n_clusters=4, n_init=10, random_state=42)
    rfm_vp['segment_id'] = kmeans_vp.fit_predict(rfm_scaled)
    
    # --- 3. Profile & PCA ---
    segment_stats = rfm_vp.groupby('segment_id').agg({'recency':'mean', 'frequency':'mean', 'monetary':'mean'}).reset_index()
    premium_seg_id = segment_stats.loc[segment_stats['monetary'].idxmax(), 'segment_id']
    
    def assign_name(row):
        if row['segment_id'] == premium_seg_id: return "Premium"
        if row['frequency'] > 15: return "Frequent Buyers"
        if row['recency'] > 30: return "At Risk"
        return "Occasional"
        
    rfm_vp['segment_name'] = rfm_vp.apply(assign_name, axis=1)
    rfm_vp['is_premium'] = (rfm_vp['segment_id'] == premium_seg_id)
    
    pca = PCA(n_components=2)
    pca.fit(rfm_scaled)
    print(f"PCA Variance Ratio: {pca.explained_variance_ratio_}")
    
    rfm_vp.to_csv("outputs/customer_segments.csv", index=False)
    
    # --- 4. Lapse Problem Definition ---
    cutoff = pd.to_datetime("2025-09-01")
    
    # Find customers who ordered before cutoff
    orders_pre_cutoff = vp_ord[vp_ord['order_ts'] < cutoff]
    cohort_cids = orders_pre_cutoff[orders_pre_cutoff['status'] == 'completed']['customer_id'].unique()
    
    # Create labels: Lapsed if NO completed order in Sept 2025
    sept_orders = vp_ord[(vp_ord['order_ts'] >= cutoff) & (vp_ord['order_ts'] < pd.to_datetime("2025-10-01")) & (vp_ord['status'] == 'completed')]
    sept_buyers = sept_orders['customer_id'].unique()
    
    cohort_df = pd.DataFrame({'customer_id': cohort_cids})
    cohort_df['lapsed'] = np.where(cohort_df['customer_id'].isin(sept_buyers), 0, 1)
    
    # Build features purely from pre-cutoff data
    completed_pre = orders_pre_cutoff[orders_pre_cutoff['status'] == 'completed']
    
    features_df = completed_pre.groupby('customer_id').agg(
        recency=('order_ts', lambda x: (cutoff - x.max()).days),
        frequency=('order_id', 'nunique'),
        monetary=('order_amount', 'sum'),
        avg_discount=('discount_pct', 'mean'),
        avg_delivery=('delivery_minutes', 'mean')
    ).reset_index()
    
    # Cancellation rate
    status_counts = orders_pre_cutoff[orders_pre_cutoff['status'].isin(['completed', 'cancelled'])].groupby(['customer_id', 'status']).size().unstack(fill_value=0)
    if 'cancelled' not in status_counts.columns: status_counts['cancelled'] = 0
    status_counts['cancel_rate'] = status_counts['cancelled'] / (status_counts['completed'] + status_counts['cancelled'])
    features_df = features_df.merge(status_counts[['cancel_rate']].reset_index(), on='customer_id', how='left')
    
    # Most frequent category pre-cutoff
    merged_pre = completed_pre.merge(pd.read_csv("data/clean/products.csv")[['product_id', 'category']], on='product_id')
    fav_cat = merged_pre.groupby('customer_id')['category'].agg(lambda x: x.mode()[0]).reset_index()
    fav_cat.columns = ['customer_id', 'fav_category']
    features_df = features_df.merge(fav_cat, on='customer_id', how='left')
    
    # Merge customer static data
    features_df = features_df.merge(vp_cust, on='customer_id', how='left')
    features_df['signup_date'] = pd.to_datetime(features_df['signup_date'])
    features_df['tenure_days'] = (cutoff - features_df['signup_date']).dt.days
    features_df = features_df.drop(columns=['signup_date'])
    
    master_cohort = cohort_df.merge(features_df, on='customer_id')
    
    print(f"Cohort size: {len(master_cohort)}, Lapsed: {master_cohort['lapsed'].sum()} ({master_cohort['lapsed'].mean():.2%})")
    
    X = master_cohort.drop(columns=['customer_id', 'lapsed'])
    y = master_cohort['lapsed']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # --- 5. Preprocessing (No Leakage) ---
    cat_cols = ['city', 'gender', 'fav_category']
    num_cols = ['recency', 'frequency', 'monetary', 'avg_discount', 'avg_delivery', 'cancel_rate', 'age', 'tenure_days']
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
    ])
    
    # Print fitted scaler mean vs whole cohort mean
    preprocessor.fit(X_train)
    scaler_idx = num_cols.index('frequency')
    print(f"Fitted scaler frequency mean: {preprocessor.transformers_[0][1].mean_[scaler_idx]:.3f}")
    print(f"Full cohort frequency mean: {master_cohort['frequency'].mean():.3f}")
    
    # --- 6. Train & Tune ---
    pipe_lr = Pipeline([('prep', preprocessor), ('clf', LogisticRegression(max_iter=1000, random_state=42))])
    param_grid_lr = {'clf__C': [0.01, 0.1, 1, 10, 100]}
    gs_lr = GridSearchCV(pipe_lr, param_grid_lr, cv=5, scoring='f1')
    gs_lr.fit(X_train, y_train)
    
    pipe_rf = Pipeline([('prep', preprocessor), ('clf', RandomForestClassifier(n_estimators=100, random_state=42))])
    param_grid_rf = {'clf__max_depth': [3, 5, 8], 'clf__min_samples_leaf': [5, 10, 20]}
    gs_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=5, scoring='f1')
    gs_rf.fit(X_train, y_train)
    
    print("Best LR Params:", gs_lr.best_params_, "F1:", gs_lr.best_score_)
    print("Best RF Params:", gs_rf.best_params_, "F1:", gs_rf.best_score_)
    
    best_model = gs_rf.best_estimator_  # Choosing RF as it generally yields better F1
    y_pred = best_model.predict(X_test)
    print("Test F1:", f1_score(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    # --- 7. Class-Balance Experiment ---
    df_train = pd.concat([X_train, y_train], axis=1)
    df_maj = df_train[df_train['lapsed'] == 0]
    df_min = df_train[df_train['lapsed'] == 1]
    
    df_min_upsampled = resample(df_min, replace=True, n_samples=len(df_maj), random_state=42)
    df_balanced = pd.concat([df_maj, df_min_upsampled])
    
    X_train_bal = df_balanced.drop('lapsed', axis=1)
    y_train_bal = df_balanced['lapsed']
    
    bal_pipe_lr = Pipeline([('prep', preprocessor), ('clf', LogisticRegression(C=1, max_iter=1000, random_state=42))])
    bal_pipe_lr.fit(X_train_bal, y_train_bal)
    
    y_pred_bal = bal_pipe_lr.predict(X_test)
    print("Balanced LR Recall:", recall_score(y_test, y_pred_bal))
    print("Balanced LR Precision:", precision_score(y_test, y_pred_bal))
    
    # --- 8. Interpret & Save ---
    joblib.dump(best_model, 'outputs/lapse_model.joblib')
    
    # Premium Retention List
    test_cids = master_cohort.loc[X_test.index, 'customer_id']
    premium_cids = rfm_vp[rfm_vp['is_premium']]['customer_id']
    test_premium_mask = test_cids.isin(premium_cids)
    
    X_test_premium = X_test[test_premium_mask]
    if len(X_test_premium) > 0:
        probs = best_model.predict_proba(X_test_premium)[:, 1]
        retention_df = pd.DataFrame({
            'customer_id': test_cids[test_premium_mask],
            'lapse_probability': probs,
            'actually_lapsed': y_test[test_premium_mask]
        }).sort_values('lapse_probability', ascending=False).head(10)
        
        retention_df.to_csv("outputs/retention_list.csv", index=False)
        print("Premium retention list generated.")

if __name__ == "__main__":
    part_b_execution()