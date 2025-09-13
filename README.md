# 🛡️ Fkae Review Detection System

A sophisticated machine learning-powered web application that detects spam and fake reviews with high accuracy, featuring an interactive 5-star rating analysis system.

## 🌟 Features

### 🤖 Core ML Capabilities
- **High Accuracy**: 89.1% spam detection accuracy with cross-validation
- **Advanced NLP**: Enhanced TF-IDF vectorization with n-gram analysis
- **Multiple Algorithms**: Optimized Logistic Regression, SVM, Random Forest, and Gradient Boosting
- **Pattern Detection**: Identifies spam patterns like excessive punctuation, ALL CAPS, and promotional language

### ⭐ Rating Analysis System
- **Interactive Star Rating**: 1-5 star rating input system
- **Sentiment Analysis**: Compares user ratings with text sentiment
- **Authenticity Scoring**: Measures rating-content consistency (0-100%)
- **Fake Review Detection**: Advanced detection of rating manipulation
- **Rating Prediction**: Estimates expected ratings based on text content

### 🎨 Modern Web Interface
- **Responsive Design**: Professional dark theme with white text
- **Navigation Tabs**: Clean, organized information layout
- **Multiple Input Methods**: Type text, upload files, or use pre-defined examples
- **Real-time Analysis**: Instant spam detection with confidence scores
- **Visual Results**: Color-coded alerts and detailed breakdown

### 📊 Advanced Analytics
- **Confidence Scoring**: Probability estimates for predictions
- **Risk Assessment**: HIGH/MEDIUM/LOW risk categorization
- **Text Processing Insights**: Shows original vs. processed text
- **Performance Metrics**: Detailed model statistics and cross-validation scores

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip (Python package installer)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/spam-detection-system.git
cd spam-detection-system
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the dataset**
   - Download the `op_spam_v1.4` dataset
   - Place it in the project root directory

4. **Train the model**
```bash
python spam_model_training.py
```

5. **Run the web application**
```bash
streamlit run streamlit_app.py
```

6. **Access the app**
   - Open your browser and go to `http://localhost:8501`

## 📁 Project Structure

```
spam-detection-system/
│
├── spam_model_training.py     # Main model training script
├── streamlit_app.py          # Web application interface
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── background.jpg            # Optional background image
│
├── op_spam_v1.4/            # Dataset directory
│   ├── negative_polarity/
│   │   ├── deceptive_from_MTurk/
│   │   └── truthful_from_Web/
│   └── positive_polarity/
│       ├── deceptive_from_MTurk/
│       └── truthful_from_Web/
│
├── spam_detection_model.pkl  # Trained model (generated)
├── model_info.json          # Model metadata (generated)
└── .venv/                   # Virtual environment (optional)
```

## 🔧 Technical Details

### Model Architecture
- **Primary Algorithm**: Logistic Regression (optimized)
- **Vectorization**: TF-IDF with enhanced parameters
  - Max features: 8,000
  - N-gram range: (1,3) - includes trigrams
  - Min document frequency: 2
  - Max document frequency: 95%
- **Cross-validation**: 5-fold CV with 87.97% ± 3.54% accuracy
- **Training Data**: 1,600 hotel reviews (balanced dataset)

### Text Preprocessing Pipeline
```python
def preprocess_text(text):
    # Convert to lowercase
    # Detect spam patterns (!!!, ???, ALL CAPS)
    # Mark URLs and email addresses
    # Tokenize numbers as NUMBER placeholders
    # Remove excessive punctuation
    # Preserve sentence structure
```

### Spam Detection Features
- Multiple exclamation marks (`!!!`)
- ALL CAPS words detection
- URL and email pattern recognition
- Promotional language identification
- Urgency indicators (`URGENT`, `LIMITED TIME`)
- Superlative overuse detection

## 📊 Dataset Information

### Source
**Opinion Spam Dataset v1.4**
- 1,600 hotel reviews from TripAdvisor
- Balanced distribution: 50% spam, 50% legitimate
- 800 positive polarity, 800 negative polarity reviews

### Data Distribution
| Category | Count | Percentage |
|----------|-------|------------|
| Deceptive Reviews | 800 | 50% |
| Truthful Reviews | 800 | 50% |
| Positive Polarity | 800 | 50% |
| Negative Polarity | 800 | 50% |

## 🎯 Usage Examples

### Basic Text Analysis
```python
# Example 1: Suspicious review
text = "AMAZING hotel!!! Best deal EVER!!! Book NOW!!!"
# Expected: High spam probability

# Example 2: Genuine review  
text = "Nice hotel with clean rooms. Staff was helpful and location convenient."
# Expected: Low spam probability
```

### Rating Analysis
1. Enter or paste review text
2. Select a star rating (1-5 stars)
3. System compares rating with text sentiment
4. Provides authenticity assessment

### File Upload
- Supports `.txt` files
- Automatically processes file content
- Displays original text alongside analysis

## 📈 Performance Metrics

### Model Performance
- **Test Accuracy**: 89.06%
- **Cross-Validation**: 87.97% ± 3.54%
- **Precision**: 89% (spam detection)
- **Recall**: 89% (spam detection)
- **F1-Score**: 89%

### System Performance
- **Processing Time**: <1 second per review
- **Memory Usage**: ~50MB for loaded model
- **Scalability**: Handles concurrent users efficiently


### App Configuration
```python
# Streamlit Settings
page_title="Spam Detection System"
page_icon="🛡️"
layout="wide"
initial_sidebar_state="collapsed"
```



## 🔍 Debugging Guide

### Common Issues

**Model Not Loading**
```
Error: Model files not found!
Solution: Run python spam_model_training.py first
```

**NLTK Data Missing**
```
Warning: NLTK data download failed
Solution: Manual download - python -m nltk.downloader stopwords punkt
```

**Low Accuracy**
```
Issue: Model accuracy below expectations
Solutions:
- Check data quality and balance
- Tune hyperparameters
- Try ensemble methods
- Increase training data
```

**Web App Not Starting**
```
Error: ModuleNotFoundError
Solution: pip install -r requirements.txt
```

### Performance Optimization
- Use virtual environment for clean dependencies
- Enable caching for model loading (`@st.cache_resource`)
- Optimize text preprocessing for large texts
- Consider model quantization for production

## 📦 Dependencies

### Core Libraries
```
streamlit>=1.28.0       # Web application framework
pandas>=1.5.0           # Data manipulation
scikit-learn>=1.3.0     # Machine learning
joblib>=1.3.0           # Model serialization
nltk>=3.8               # Natural language processing
numpy>=1.24.0           # Numerical computing
```

### Optional Libraries
```
matplotlib>=3.6.0       # Plotting (for analysis)
seaborn>=0.12.0        # Statistical visualization
plotly>=5.15.0         # Interactive plots
```

## 🚀 Deployment

### Local Development
```bash
streamlit run streamlit_app.py --server.port 8501
```

### Production Deployment

**Streamlit Cloud**
1. Push to GitHub repository
2. Connect to Streamlit Cloud
3. Deploy directly from repository


<img width="1792" height="830" alt="image" src="https://github.com/user-attachments/assets/3c848aa8-6d3a-49bb-9e93-ee5394e08c72" />

Start detecting spam like a pro! 🕵️‍♂️✨"# Fake-review-Detection" 
