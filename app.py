
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Product Recommendation Engine", page_icon="🔎", layout="wide")

# ---- Inject Custom CSS ----
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    h1 { color: #1f77b4; }
    h2, h3, h4 { color: #2ca02c; }
    div.stAlert > div { background-color: #e3f2fd; }
    div[data-testid="stDataFrame"] {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 10px;
    }
    .sidebar .sidebar-content { background-color: #f1f3f5; }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5em 1em;
        font-size: 1em;
        transition: background-color 0.3s;
    }
    .stButton>button:hover { background-color: #45a049; }
    </style>
""", unsafe_allow_html=True)

# ---- Data Loading ----
data = pd.read_csv("customer_data_with_clusters.csv")
columns_to_drop = ['Size', 'Sub_Total', 'Shipping_Fee','Total_Amount','Bill_To_City','Type_Of_Sale','Year','Month_Name']
data = data.drop(columns=columns_to_drop)

columns = list(data.columns)
columns.remove('Cluster')
columns.insert(2, 'Cluster')
data = data[columns]

# ---- Preprocessing ----
data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d", errors="coerce")
data["Product_Description"] = data["Product_Description"].astype(str)

# ---- App Title ----
st.title("🔎 Product Recommendation Engine")
st.write("Explore four different recommendation models to better understand customer preferences and product opportunities.")
st.markdown("---")

# ---- Sidebar ----
st.sidebar.header("🛠️ Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose Recommendation Model", ["Collaborative Filtering", "Content-Based", "Hybrid", "Slow-Moving Products"]
)

# ---- Prepare models ----
@st.cache_data
def prepare_models():
    user_item_matrix = data.pivot_table(
        index="Customer_ID",
        columns="Product_Description",
        values="Quantity_Sold",
        fill_value=0,
    )
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index
    )
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(data["Product_Description"].unique())
    product_similarity = cosine_similarity(tfidf_matrix)
    product_similarity_df = pd.DataFrame(
        product_similarity,
        index=data["Product_Description"].unique(),
        columns=data["Product_Description"].unique(),
    )
    return user_item_matrix, user_similarity_df, product_similarity_df

user_item_matrix, user_similarity_df, product_similarity_df = prepare_models()

# ---- Recommendations Placeholder ----
recommendations_df = pd.DataFrame()

# ---- Recommendation Functions ----
def collaborative_recommend(customer_id, top_n=5):
    if customer_id not in user_similarity_df.index:
        return []
    similar_users = user_similarity_df[customer_id].sort_values(ascending=False).iloc[1:6].index
    recommendations = data[data["Customer_ID"].isin(similar_users)]
    recommendations = recommendations.groupby("Product_Description").sum(numeric_only=True)
    return recommendations.sort_values("Quantity_Sold", ascending=False).head(top_n).index.tolist()

def content_based_recommend(product_name, top_n=5):
    if product_name not in product_similarity_df.index:
        return []
    similar_products = product_similarity_df.loc[product_name].sort_values(ascending=False).iloc[1:top_n+1]
    return list(similar_products.index)

def hybrid_recommend(customer_id, product_name, top_n=5, weight_cf=0.6, weight_cb=0.4):
    hybrid_scores = {}
    if customer_id in user_similarity_df.index:
        similar_users = user_similarity_df[customer_id].sort_values(ascending=False).iloc[1:6].index
        collab_recs = data[data["Customer_ID"].isin(similar_users)]
        collab_scores = collab_recs.groupby("Product_Description").sum(numeric_only=True)["Quantity_Sold"]
        if len(collab_scores) > 0:
            max_cf = collab_scores.max()
            collab_scores = collab_scores / max_cf if max_cf > 0 else collab_scores
    else:
        collab_scores = pd.Series()

    if product_name in product_similarity_df.index:
        content_scores = product_similarity_df.loc[product_name]
        max_cb = content_scores.max()
        content_scores = content_scores / max_cb if max_cb > 0 else content_scores
    else:
        content_scores = pd.Series()

    all_products = set(collab_scores.index).union(set(content_scores.index))
    for product in all_products:
        cf_score = collab_scores.get(product, 0)
        cb_score = content_scores.get(product, 0)
        hybrid_scores[product] = (weight_cf * cf_score) + (weight_cb * cb_score)

    hybrid_scores.pop(product_name, None)
    return [p for p, _ in sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]]

def slow_moving_products(top_n=5):
    product_sales = data.groupby("Product_Description").sum(numeric_only=True)["Quantity_Sold"]
    return product_sales.sort_values(ascending=True).head(top_n).index.tolist()

# ---- Main App Logic ----
with st.container():
    if model_choice == "Collaborative Filtering":
        st.subheader("👥 Collaborative Filtering Recommendations")
        customer_id = st.selectbox("Select Customer ID", options=data["Customer_ID"].unique())
        if st.button("🎯 Get Recommendations"):
            with st.spinner("Finding best matches..."):
                recommendations = collaborative_recommend(customer_id)
                if recommendations:
                    st.success("Top Recommendations for You:")
                    for i, product in enumerate(recommendations, 1):
                        st.markdown(f"**{i}. {product}**")
                    recommendations_df = data[data["Product_Description"].isin(recommendations)]

                    # 📌 Show customer cluster
                    customer_cluster = data[data["Customer_ID"] == customer_id]["Cluster"].values
                    if customer_cluster.size > 0:
                        st.info(f"**Selected Customer belongs to Cluster:** {customer_cluster[0]}")

                    # 📌 Cluster Distribution
                    cluster_counts = recommendations_df["Cluster"].value_counts()
                    cluster_df = cluster_counts.rename_axis('Cluster').reset_index(name='Count')
                    st.subheader("📋 Cluster Distribution in Recommendations")
                    st.dataframe(
                        cluster_df.style.background_gradient(cmap='Blues'),
                        use_container_width=True
                    )
                else:
                    st.warning("No recommendations found for this customer.")

    elif model_choice == "Content-Based":
        st.subheader("🛍️ Content-Based Recommendations")
        product_name = st.selectbox("Select a Product", options=data["Product_Description"].unique())
        if st.button("🎯 Get Recommendations"):
            with st.spinner("Searching for similar products..."):
                recommendations = content_based_recommend(product_name)
                if recommendations:
                    st.success("You might also like:")
                    for i, product in enumerate(recommendations, 1):
                        st.markdown(f"**{i}. {product}**")
                    recommendations_df = data[data["Product_Description"].isin(recommendations)]

                    cluster_counts = recommendations_df["Cluster"].value_counts()
                    cluster_df = cluster_counts.rename_axis('Cluster').reset_index(name='Count')
                    st.subheader("📋 Cluster Distribution in Recommendations")
                    st.dataframe(
                        cluster_df.style.background_gradient(cmap='Blues'),
                        use_container_width=True
                    )
                else:
                    st.warning("Product not found.")

    elif model_choice == "Hybrid":
        st.subheader("🔀 Hybrid Recommendations")
        col1, col2 = st.columns(2)
        with col1:
            customer_id = st.selectbox("Select Customer ID", options=data["Customer_ID"].unique())
        with col2:
            product_name = st.selectbox("Select Product", options=data["Product_Description"].unique())

        cf_weight = st.slider("Collaborative Filtering Weight", 0.0, 1.0, 0.6)
        cb_weight = st.slider("Content-Based Weight", 0.0, 1.0, 0.4)

        if st.button("🎯 Get Recommendations"):
            with st.spinner("Blending models..."):
                recommendations = hybrid_recommend(customer_id, product_name, weight_cf=cf_weight, weight_cb=cb_weight)
                if recommendations:
                    st.success("Top Hybrid Recommendations:")
                    for i, product in enumerate(recommendations, 1):
                        st.markdown(f"**{i}. {product}**")
                    recommendations_df = data[data["Product_Description"].isin(recommendations)]

                    cluster_counts = recommendations_df["Cluster"].value_counts()
                    cluster_df = cluster_counts.rename_axis('Cluster').reset_index(name='Count')
                    st.subheader("📋 Cluster Distribution in Recommendations")
                    st.dataframe(
                        cluster_df.style.background_gradient(cmap='Blues'),
                        use_container_width=True
                    )
                else:
                    st.warning("No hybrid recommendations found.")

    elif model_choice == "Slow-Moving Products":
        st.subheader("🐌 Slow-Moving Products")
        top_n = st.slider("Select Number of Slow-Moving Products", min_value=1, max_value=10, value=5)
        if st.button("🎯 Get Slow-Moving Products"):
            with st.spinner("Identifying slow movers..."):
                slow_movers = slow_moving_products(top_n)
                if slow_movers:
                    st.success("Consider promoting these slow-moving products:")
                    for i, product in enumerate(slow_movers, 1):
                        st.markdown(f"**{i}. {product}**")
                    recommendations_df = data[data["Product_Description"].isin(slow_movers)]

                    cluster_counts = recommendations_df["Cluster"].value_counts()
                    cluster_df = cluster_counts.rename_axis('Cluster').reset_index(name='Count')
                    st.subheader("📋 Cluster Distribution in Recommendations")
                    st.dataframe(
                        cluster_df.style.background_gradient(cmap='Blues'),
                        use_container_width=True
                    )
                else:
                    st.warning("No slow-moving products found.")

st.markdown("---")

# 📂 Data Preview
st.sidebar.header("📂 Data Preview")
if st.sidebar.checkbox("Show raw data related to recommendations"):
    st.subheader("📋 Raw Data Related to Your Recommendations")
    if not recommendations_df.empty:
        st.dataframe(recommendations_df, use_container_width=True)
    else:
        st.info("No recommendations generated yet. Please select and generate recommendations first.")
