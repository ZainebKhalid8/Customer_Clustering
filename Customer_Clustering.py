import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load Trained KMeans Model
kmeans = pickle.load(open("kmeans.pkl", 'rb'))

# Simple clustering function
def clustering(age, avg_spend, visit_per_week, promotion_interest):
    new_customer = np.array([[age, avg_spend, visit_per_week, promotion_interest]])
    predicted_cluster = kmeans.predict(new_customer)
    return predicted_cluster[0], new_customer

# Streamlit app here==========================================
st.title("Customer Clustering App")
st.write("Enter the customer details:")

# User input (side by side inputs)
# row 1 with column 2
col1, col2 = st.columns(2)
with col1:
    st.subheader("Customer Age")
    age = st.number_input("Age", min_value=18, max_value=100, value=40)

with col2:
    st.subheader("Customer Spent Time")
    avg_spend = st.number_input("Average Spend", min_value=0.0, max_value=1000.0, value=30.0)

# row 2 with column 2
col1, col2 = st.columns(2)
with col1:
    st.subheader("Visits per week")
    visit_per_week = st.number_input("Visits per Week", min_value=0, max_value=20, value=4)

with col2:
    st.subheader("Promotion Interest")
    promotion_interest = st.number_input("Promotion Interest", min_value=0, max_value=10, value=7)

# Predict button
if st.button("Predict Cluster"):
    cluster_label, new_customer = clustering(age, avg_spend, visit_per_week, promotion_interest)
    st.success(f'The customer belongs to the "{["Daily", "Weekend", "Promotion"][cluster_label]}" cluster.')

    # Plotting the clusters and new customer
    fig, ax = plt.subplots()
    cluster_data = kmeans.cluster_centers_
    
    # Scatter plot of the original clusters
    sns.scatterplot(x=cluster_data[:, 0], y=cluster_data[:, 1], s=200, marker='X', color='red', label='Cluster Centers', ax=ax)
    ax.scatter(new_customer[0][0], new_customer[0][1], color='blue', s=150, edgecolor='k', marker='o', label='New Customer')
    ax.set_xlabel('Age')
    ax.set_ylabel('Average Spend')
    ax.set_title('Customer Segmentation')
    ax.legend()
    st.pyplot(fig)

    # Additional Visualizations ======================================
    # 1. Line Plot: Age vs. Avg Spend for cluster centers
    fig2, ax2 = plt.subplots()
    ax2.plot(cluster_data[:, 0], cluster_data[:, 1], marker='o', linestyle='-', color='green', label='Age vs. Avg Spend')
    ax2.set_xlabel('Age')
    ax2.set_ylabel('Average Spend')
    ax2.set_title('Line Plot - Age vs. Average Spend')
    ax2.legend()
    st.pyplot(fig2)

    # 2. Scatter Plot: Visits per Week vs. Promotion Interest
    fig3, ax3 = plt.subplots()
    sns.scatterplot(x=cluster_data[:, 2], y=cluster_data[:, 3], s=200, marker='X', color='purple', label='Cluster Centers', ax=ax3)
    ax3.scatter(new_customer[0][2], new_customer[0][3], color='orange', s=150, edgecolor='k', marker='o', label='New Customer')
    ax3.set_xlabel('Visits per Week')
    ax3.set_ylabel('Promotion Interest')
    ax3.set_title('Scatter Plot - Visits per Week vs. Promotion Interest')
    ax3.legend()
    st.pyplot(fig3)

    # 3. Distribution Plot: Distribution of Age
    fig4, ax4 = plt.subplots()
    sns.histplot(cluster_data[:, 0], bins=10, kde=True, color='skyblue', ax=ax4)
    ax4.axvline(new_customer[0][0], color='red', linestyle='--', linewidth=2, label='New Customer Age')
    ax4.set_xlabel('Age')
    ax4.set_title('Distribution of Age')
    ax4.legend()
    st.pyplot(fig4)
