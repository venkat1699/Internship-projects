# Import necessary libraries
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the initial dataset
initial_df = None

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global initial_df

    # Handle file upload
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', message='No selected file')

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Read the uploaded file into a DataFrame
        initial_df = pd.read_excel(file_path)

        # Preprocess the initial DataFrame...
        initial_df['Income'] = initial_df['Income'].fillna(initial_df['Income'].median())
        initial_df = initial_df.drop(columns=["Z_CostContact", "Z_Revenue"], axis=1)
        initial_df['Education'] = initial_df['Education'].replace(['PhD', '2n Cycle', 'Graduation', 'Master'], 'Post Graduate')
        initial_df['Education'] = initial_df['Education'].replace(['Basic'], 'Under Graduate')
        initial_df['Marital_Status'] = initial_df['Marital_Status'].replace(['Married', 'Together'], 'Relationship')
        initial_df['Marital_Status'] = initial_df['Marital_Status'].replace(['Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd'], 'Single')
        initial_df['Children'] = initial_df['Kidhome'] + initial_df['Teenhome']
        initial_df['Expenditure'] = initial_df['MntWines'] + initial_df['MntFruits'] + initial_df['MntMeatProducts'] + initial_df['MntFishProducts'] + initial_df['MntSweetProducts'] + initial_df['MntGoldProds']
        initial_df['Overall_Accepted_Cmp'] = initial_df['AcceptedCmp1'] + initial_df['AcceptedCmp2'] + initial_df['AcceptedCmp3'] + initial_df['AcceptedCmp4'] + initial_df['AcceptedCmp5']
        initial_df['NumTotalPurchases'] = initial_df['NumWebPurchases'] + initial_df['NumCatalogPurchases'] + initial_df['NumStorePurchases'] + initial_df['NumDealsPurchases']
        initial_df['Customer_Age'] = (pd.Timestamp('now').year) - initial_df['Year_Birth']
        col_del = ["Year_Birth","ID","AcceptedCmp1" , "AcceptedCmp2", "AcceptedCmp3" , "AcceptedCmp4","AcceptedCmp5","NumWebVisitsMonth", "NumWebPurchases","NumCatalogPurchases","NumStorePurchases","NumDealsPurchases" , "Kidhome", "Teenhome","MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
        initial_df = initial_df.drop(columns=col_del, axis=1)
        initial_df["Dt_Customer"] = pd.to_datetime(initial_df["Dt_Customer"])
        dates = [i.date() for i in initial_df["Dt_Customer"]]
        days = [(max(dates) - i).days for i in dates]
        initial_df["Customer_Shop_Days"] = days
        initial_df = initial_df.drop(['Dt_Customer','Recency','Complain','Response'], axis=1)
        initial_df = initial_df[initial_df['Income'] < 70000]
        initial_df = initial_df[initial_df['Customer_Age'] < 90]
        initial_df = initial_df[initial_df['Expenditure'] < 1050]
        initial_df = initial_df[initial_df['NumTotalPurchases'] < 21.1]
        label_encoder = LabelEncoder()
        initial_df['Education'] = label_encoder.fit_transform(initial_df['Education'])
        initial_df['Marital_Status'] = label_encoder.fit_transform(initial_df['Marital_Status'])
        scaler = StandardScaler()
        col_scale = ['Income', 'Children', 'Expenditure','Overall_Accepted_Cmp', 'NumTotalPurchases', 'Customer_Age', 'Customer_Shop_Days']
        initial_df[col_scale] = scaler.fit_transform(initial_df[col_scale])

    # Handle new column data input
    education = request.form.get('education')
    marital_status = request.form.get('marital_status')
    income = request.form.get('income')
    children = request.form.get('children')
    expenditure = request.form.get('expenditure')
    overall_accepted_cmp = request.form.get('overall_accepted_cmp')
    num_total_purchases = request.form.get('num_total_purchases')
    customer_age = request.form.get('customer_age')
    customer_shop_days = request.form.get('customer_shop_days')

    # Check if any required form input is missing
    if None in [education, marital_status, income, children, expenditure, overall_accepted_cmp, num_total_purchases, customer_age, customer_shop_days]:
        return render_template('index.html', message='Some form inputs are missing or invalid')

    # Convert inputs to appropriate types
    income = float(income)
    children = int(children)
    expenditure = float(expenditure)
    overall_accepted_cmp = int(overall_accepted_cmp)
    num_total_purchases = int(num_total_purchases)
    customer_age = int(customer_age)
    customer_shop_days = int(customer_shop_days)

    # Create a DataFrame with the new input data
    new_data = pd.DataFrame({
        'Education': [education],
        'Marital_Status': [marital_status],
        'Income': [income],
        'Children': [children],
        'Expenditure': [expenditure],
        'Overall_Accepted_Cmp': [overall_accepted_cmp],
        'NumTotalPurchases': [num_total_purchases],
        'Customer_Age': [customer_age],
        'Customer_Shop_Days': [customer_shop_days]
    })

    # Combine initial DataFrame with new input data
    updated_df = pd.concat([initial_df, new_data])

    # # Preprocess updated DataFrame...
   # Encode categorical variables if needed
    
    updated_df['Marital_Status'] = updated_df['Marital_Status'].astype(str)
    updated_df['Education'] = updated_df['Education'].astype(str)
    
    label_encoder = LabelEncoder()
    updated_df['Education'] = label_encoder.fit_transform(updated_df['Education'])
    updated_df['Marital_Status'] = label_encoder.fit_transform(updated_df['Marital_Status'])

    scaler = StandardScaler()
    col_scale = ['Income', 'Children', 'Expenditure','Overall_Accepted_Cmp', 'NumTotalPurchases', 'Customer_Age', 'Customer_Shop_Days']
    updated_df[col_scale] = scaler.fit_transform(updated_df[col_scale])
    new_df=updated_df.copy()

    # Calculate silhouette score
    silhouette_score_avg , Predictions ,plot_path ,plot_path_pi= calculate_silhouette_score(new_df)
    

    return render_template('result.html', silhouette_score=silhouette_score_avg,Predictions=Predictions ,plot_path=plot_path,plot_path_pi=plot_path_pi)

def calculate_silhouette_score(df):
    # Preprocess the data, apply PCA, perform Agglomerative Clustering, and calculate silhouette score
    pca = PCA(n_components=3)
    pca.fit(df)
    PCA_ds = pd.DataFrame(pca.transform(df), columns=(["col1","col2", "col3"]))
    #Initiating the Agglomerative Clustering model
    AC = AgglomerativeClustering(n_clusters=2)
    # fit model and predict clusters
    yhat_AC = AC.fit_predict(PCA_ds)
    PCA_ds["AgglomerativeClustering"] = yhat_AC
    #Adding the Clusters feature to the original dataframe.
    df["Cluster_Agglo"]= yhat_AC + 1
    # Datapoints present in each cluster
    Predictions=PCA_ds["AgglomerativeClustering"].value_counts()        
    silhouette_avg  = silhouette_score(PCA_ds, yhat_AC)
    # Return the silhouette score
    sns.scatterplot(x=df['Expenditure'], y=df['Income'], hue=df['Cluster_Agglo'], palette='dark')
    plt.xlabel("Expenditure")
    plt.ylabel("Income")
    plt.title("Scatter Plot of Expenditure vs. Income")
     # Save the plot to a file
    plot_path = "static/plot.png"
    plt.savefig(plot_path)
        
    # Close the plot to free up memory
    plt.close()

    cluster_counts = df["Cluster_Agglo"].value_counts()
    plt.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%')
    plt.title("Distribution Of The Clusters")
        
        # Save the plot to a file
    plot_path_pi = "static/pie_chart.png"
    plt.savefig(plot_path_pi)
        
        # Close the plot to free up memory
    plt.close()
    return silhouette_avg ,Predictions, plot_path ,plot_path_pi# Placeholder value, replace with actual calculation

if __name__ == '__main__':
    app.run(debug=True)
