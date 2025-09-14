# Wine-Quality-Prediction-System

A comprehensive machine learning-based system designed to predict wine quality based on various physicochemical properties. This project includes multiple interfaces: a web application, a desktop GUI application, and a Jupyter notebook for analysis.

## 🍷 Project Overview

The Wine Quality Prediction System uses machine learning algorithms to evaluate wine quality based on 11 physicochemical features. The system classifies wines into two categories: "Good Quality Wine" (score ≥ 7) and "Average Quality Wine" (score < 7).

## 📊 Dataset Information

- **Source**: UCI Machine Learning Repository
- **Dataset**: Wine Quality Dataset (Red Wine)
- **Samples**: 1,599 wine samples
- **Features**: 11 numerical attributes
- **Target Variable**: Wine quality score (0-10 scale)

### Dataset Features

| Feature | Description | Data Type |
|---------|-------------|-----------|
| Fixed Acidity | Concentration of non-volatile acids (e.g., tartaric acid) | Float |
| Volatile Acidity | Concentration of volatile acids (e.g., acetic acid) | Float |
| Citric Acid | Amount of citric acid present (adds freshness) | Float |
| Residual Sugar | Amount of sugar left after fermentation | Float |
| Chlorides | Salt content in the wine | Float |
| Free Sulfur Dioxide | SO₂ available to prevent microbial growth | Float |
| Total Sulfur Dioxide | Total SO₂ (free + bound), used as a preservative | Float |
| Density | Mass per unit volume, affects wine body | Float |
| pH | Measure of acidity or alkalinity | Float |
| Sulphates | Sulfur compounds contributing to antimicrobial properties | Float |
| Alcohol | Alcohol content (% volume) | Float |

## 🚀 Features

- **Multiple Interfaces**: Web app, Desktop GUI, and Jupyter notebook
- **Machine Learning Models**: Multiple algorithms tested and compared
- **Data Preprocessing**: Standardization and feature scaling
- **Interactive Prediction**: Real-time wine quality prediction
- **Visualization**: Comprehensive EDA and model performance analysis

## 🛠️ Technologies Used

- **Python 3.x**
- **Machine Learning Libraries**:
  - scikit-learn
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - xgboost
- **Web Framework**: Flask
- **GUI Framework**: Tkinter
- **Data Visualization**: Matplotlib, Seaborn
- **Model Persistence**: Pickle

## 📁 Project Structure

```
05_Project1/
├── Wine Quality Prediction System.py    # Main ML pipeline and model training
├── web_app.py                          # Flask web application
├── GUI_app.py                          # Tkinter desktop application
├── Wine Quality Prediction System.ipynb # Jupyter notebook for analysis
├── winequality-red.csv                 # Dataset
├── wine_model                          # Trained model (pickle file)
├── wine_sc                             # StandardScaler (pickle file)
├── templates/
│   ├── index.html                      # Web app input form
│   └── predict.html                    # Web app results page
├── static/
│   └── image.png                       # Static assets
├── image.png                           # Background image for GUI
└── README.md                           # This file
```

## 🔧 Installation & Setup

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Required Libraries

Install the required packages using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost flask pillow
```

### Clone the Repository

```bash
git clone <repository-url>
cd 05_Project1
```

## 🎯 Usage

### 1. Web Application

Run the Flask web application:

```bash
python web_app.py
```

Open your browser and navigate to `http://localhost:5000` to access the web interface.

### 2. Desktop GUI Application

Run the Tkinter desktop application:

```bash
python GUI_app.py
```

This will open a desktop window with input fields for all wine parameters.

### 3. Jupyter Notebook Analysis

Open the Jupyter notebook for detailed analysis:

```bash
jupyter notebook "Wine Quality Prediction System.ipynb"
```

## 🤖 Machine Learning Pipeline

### Data Preprocessing
1. **Data Cleaning**: Removed duplicates and handled missing values
2. **Feature Engineering**: Created binary classification target (good quality ≥ 7)
3. **Feature Scaling**: Applied StandardScaler for normalization
4. **Train-Test Split**: 80% training, 20% testing

### Model Training & Evaluation

The following models were trained and evaluated:

| Model | Accuracy Score |
|-------|----------------|
| Logistic Regression | 90% |
| Support Vector Classifier | 89% |
| Random Forest | 88% |
| K-Nearest Neighbors | 88% |
| XGBoost | 88% |
| Gaussian Naive Bayes | 86% |
| Decision Tree | 83% |

**Selected Model**: Logistic Regression (90% accuracy)

### Feature Importance

The most important features for wine quality prediction are:
1. Alcohol content
2. Volatile acidity
3. Sulphates
4. Total sulfur dioxide
5. Density

## 📈 Model Performance

- **Accuracy**: 90%
- **Classification**: Binary (Good Quality vs Average Quality)
- **Cross-validation**: 5-fold cross-validation used
- **Feature Selection**: ExtraTreesClassifier for feature importance

## 🔍 Exploratory Data Analysis

The notebook includes comprehensive EDA:
- Univariate analysis (histograms, distributions)
- Bivariate analysis (correlation heatmaps, pair plots)
- Multivariate analysis (violin plots, box plots)
- Feature importance analysis
- Data quality assessment

## 🌐 Web Application Features

- **Input Form**: User-friendly interface for entering wine parameters
- **Real-time Prediction**: Instant quality prediction
- **Responsive Design**: Clean and modern UI
- **Result Display**: Clear quality classification output

## 🖥️ Desktop GUI Features

- **Interactive Interface**: Tkinter-based desktop application
- **Background Image**: Visual enhancement with wine-themed background
- **Input Validation**: Form validation for all parameters
- **Instant Results**: Real-time prediction display

## 📝 Input Parameters

To use the prediction system, provide values for these 11 parameters:

1. **Fixed Acidity** (g/dm³)
2. **Volatile Acidity** (g/dm³)
3. **Citric Acid** (g/dm³)
4. **Residual Sugar** (g/dm³)
5. **Chlorides** (g/dm³)
6. **Free Sulfur Dioxide** (mg/dm³)
7. **Total Sulfur Dioxide** (mg/dm³)
8. **Density** (g/cm³)
9. **pH** (0-14 scale)
10. **Sulphates** (g/dm³)
11. **Alcohol** (% volume)

## 🎯 Prediction Output

The system provides binary classification:
- **Good Quality Wine**: Quality score ≥ 7
- **Average Quality Wine**: Quality score < 7

## 🔧 Customization

### Adding New Models

To add new machine learning models:

1. Import the model in the main Python file
2. Train the model using the same preprocessing pipeline
3. Compare performance with existing models
4. Update the model selection if better performance is achieved

### Modifying the Web Interface

Edit the HTML templates in the `templates/` directory:
- `index.html`: Input form interface
- `predict.html`: Results display page

### Updating the GUI

Modify `GUI_app.py` to customize the desktop application interface.

## 📊 Future Enhancements

- [ ] Add more wine datasets (white wine, sparkling wine)
- [ ] Implement regression models for exact quality score prediction
- [ ] Add data visualization dashboard
- [ ] Implement model retraining pipeline
- [ ] Add confidence scores for predictions
- [ ] Create API endpoints for external integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👥 Authors

- **Data Science & Machine Learning Project**
- **Project 1**: Wine Quality Prediction System

## 📞 Support

For questions or support, please open an issue in the repository or contact the development team.

---

**Note**: This project is for educational and research purposes. The wine quality predictions should not be used as the sole basis for commercial wine quality assessment.
