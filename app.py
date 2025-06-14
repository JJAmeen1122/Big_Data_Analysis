from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
import numpy as np
from io import StringIO
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive Agg backend
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import base64
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max upload
app.config['MAX_MEMORY_BUFFER'] = 1024 * 1024 * 1024  # 1GB memory buffer
app.config['CHUNK_SIZE'] = 100000  # Process data in chunks of 100k rows

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to store data between requests
global_df = None
global_results = {}

def save_plot_to_base64():
    """Save the current matplotlib plot to a base64 encoded string."""
    import io
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return image_base64

def process_large_file(filepath):
    """Process large files in chunks to avoid memory issues"""
    try:
        if filepath.endswith('.csv'):
            try:
                chunks = pd.read_csv(filepath, chunksize=app.config['CHUNK_SIZE'], encoding='utf-8')
                df = pd.concat(chunks)
            except UnicodeDecodeError:
                try:
                    chunks = pd.read_csv(filepath, chunksize=app.config['CHUNK_SIZE'], encoding='latin1')
                    df = pd.concat(chunks)
                except UnicodeDecodeError:
                    chunks = pd.read_csv(filepath, chunksize=app.config['CHUNK_SIZE'], encoding='ISO-8859-1')
                    df = pd.concat(chunks)
        elif filepath.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format")
        
        return df
    except Exception as e:
        raise ValueError(f"Error processing file: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            
            try:
                df = process_large_file(filename)
                global global_df
                global_df = df
                
                if len(df) > 100000:
                    sample_df = df.sample(10000)
                    description = sample_df.describe().to_html(classes='table table-striped table-bordered')
                    note = "<div class='alert alert-info'>Note: Showing statistics for 10,000 row sample</div>"
                else:
                    description = df.describe().to_html(classes='table table-striped table-bordered')
                    note = ""
                
                columns = list(df.columns)
                dtypes = df.dtypes.to_frame().to_html(classes='table table-striped table-bordered')
                
                return render_template('index.html', 
                                    description=description, 
                                    columns=columns, 
                                    dtypes=dtypes,
                                    filename=file.filename,
                                    note=note)
            except Exception as e:
                return render_template('index.html', error=f"Error processing file: {str(e)}")
    
    return render_template('index.html')



@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    global global_df, global_results
    
    if global_df is None:
        return redirect(url_for('index'))
    
    target_col = request.form.get('target')
    features = request.form.getlist('features')
    algorithm = request.form.get('algorithm')
    test_size = float(request.form.get('test_size', 0.2))
    chart_type = request.form.get('chart_type')
    sample_size = int(request.form.get('sample_size', 10000))
    
    try:
        df = global_df.copy()
        if len(df) > 50000:
            df = df.sample(min(sample_size, len(df)))
            note = f"<div class='alert alert-info'>Note: Using sample of {len(df)} rows for analysis</div>"
        else:
            note = ""
        
        le = LabelEncoder()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = le.fit_transform(df[col].astype(str))
        
        X = df[features]
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = None
        if algorithm == 'knn':
            model = KNeighborsClassifier(n_jobs=-1)
        elif algorithm == 'kmeans':
            model = KMeans(n_clusters=len(y.unique()), random_state=42)
        elif algorithm == 'naive_bayes':
            model = GaussianNB()
        elif algorithm == 'logistic_regression':
            model = LogisticRegression(max_iter=1000, n_jobs=-1)
        elif algorithm == 'decision_tree':
            model = DecisionTreeClassifier()
        elif algorithm == 'svm':
            model = SVC()
        
        if algorithm == 'kmeans':
            clusters = model.fit_predict(X_train)
            y_pred = model.predict(X_test)
            accuracy = None
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
        
        chart_images = []
        plt.switch_backend('Agg')
        rcParams.update({'figure.max_open_warning': 0})
        
        if chart_type == 'bar':
            plt.figure(figsize=(10, 6))
            sns.countplot(x=y_pred, palette='viridis')
            plt.title('Prediction Distribution')
            chart_img = save_plot_to_base64()
            chart_images.append(('Bar Chart', chart_img))
        elif chart_type == 'pie':
            plt.figure(figsize=(8, 8))
            pd.Series(y_pred).value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette('bright'))
            plt.title('Prediction Distribution')
            chart_img = save_plot_to_base64()
            chart_images.append(('Pie Chart', chart_img))
        elif chart_type == 'histogram':
            plt.figure(figsize=(10, 6))
            plt.hist(y_pred, bins=20, color='#4e79a7')
            plt.title('Prediction Distribution')
            chart_img = save_plot_to_base64()
            chart_images.append(('Histogram', chart_img))
        elif chart_type == 'scatter':
            if X.shape[1] >= 2:
                plt.figure(figsize=(10, 6))
                plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_pred, alpha=0.5, cmap='viridis')
                plt.title('Scatter Plot with Predictions')
                plt.xlabel(features[0])
                plt.ylabel(features[1])
                chart_img = save_plot_to_base64()
                chart_images.append(('Scatter Plot', chart_img))
        
        if algorithm != 'kmeans':
            plt.figure(figsize=(8, 6))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            cm_img = save_plot_to_base64()
            chart_images.append(('Confusion Matrix', cm_img))
        
        global_results = {
            'algorithm': algorithm.replace('_', ' ').title(),
            'accuracy': accuracy,
            'features': features,
            'target': target_col,
            'charts': chart_images,
            'classification_report': classification_report(y_test, y_pred) if algorithm != 'kmeans' else None,
            'note': note
        }
        
        return redirect(url_for('result'))
    
    except Exception as e:
        return render_template('index.html', error=f"Error during analysis: {str(e)}")

@app.route('/result')
def result():
    return render_template('result.html', results=global_results)

@app.route('/top_videos', methods=['GET'])
def top_videos():
    global global_df
    
    if global_df is None:
        return redirect(url_for('index'))
    
    try:
        required_columns = {'video_id', 'likes'}
        if not required_columns.issubset(global_df.columns):
            missing = required_columns - set(global_df.columns)
            return render_template('error.html', 
                                error=f"Dataset is missing required columns: {', '.join(missing)}")
        
        # Get top videos - group by video_id to get max likes per video
        top_videos = (global_df.groupby('video_id', as_index=False)
                      .agg({'likes': 'max'})
                      .nlargest(10, 'likes'))
        
        videos_list = top_videos.to_dict('records')
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        sns.barplot(x='video_id', y='likes', data=top_videos, palette='rocket')
        plt.title('Top 10 Videos by Likes (Unique Videos)')
        plt.xticks(rotation=45)
        chart_img = save_plot_to_base64()
        plt.close()  # Close the figure to prevent memory leaks
        
        return render_template('top_videos.html', videos=videos_list, chart_img=chart_img)
    
    except Exception as e:
        return render_template('error.html', error=f"Error processing top videos: {str(e)}")


@app.route('/top_disliked_videos', methods=['GET'])
def top_disliked_videos():
    global global_df
    
    if global_df is None:
        return redirect(url_for('index'))
    
    try:
        # Check for required columns including 'likes' for the ratio calculation
        required_columns = {'video_id', 'dislikes', 'likes', 'title'}
        if not required_columns.issubset(global_df.columns):
            missing = required_columns - set(global_df.columns)
            return render_template('error.html', 
                                error=f"Dataset is missing required columns: {', '.join(missing)}")
        
        # Get top disliked videos with all needed information
        top_disliked = (global_df.groupby('video_id', as_index=False)
                        .agg({
                            'title': 'first',
                            'dislikes': 'max',
                            'likes': 'max'
                        })
                        .nlargest(10, 'dislikes'))
        
        # Calculate total dislikes for percentage
        total_dislikes = global_df['dislikes'].sum()
        
        # Prepare data for template
        videos_list = top_disliked.to_dict('records')
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        sns.barplot(x='title', y='dislikes', 
                   data=top_disliked, 
                   palette='mako_r')  # Reversed palette for dislikes
        plt.title('Top 10 Most Disliked Videos')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        chart_img = save_plot_to_base64()
        plt.close()
        
        return render_template('top_disliked_videos.html', 
                            videos=videos_list,
                            chart_img=chart_img,
                            total_dislikes=total_dislikes)
    
    except Exception as e:
        return render_template('error.html', 
                            error=f"Error processing top disliked videos: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)