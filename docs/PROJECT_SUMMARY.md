# 📱 Mobile Price Predictor - Project Summary

## 🎯 Project Overview

This is a complete **Smartphone Price Predictor** web application that predicts smartphone price ranges using machine learning. It includes a modern frontend, a Flask backend, and optional AWS SageMaker integration for cloud training and deployment.

## ✅ What's Included

### Frontend (Modern UI)
- ✨ **HTML5** - Responsive, semantic markup
- 🎨 **CSS3** - Modern design with gradients, animations, and responsive layout
- ⚡ **JavaScript** - Form validation, API calls, and dynamic UI updates
- 📱 **Mobile-friendly** - Works on all devices

### Backend (Flask)
- 🔧 **Flask Application** (`app.py`) - Main application server
- 🔌 **RESTful API** - `/predict`, `/health`, `/endpoint-status` endpoints
- 🛡️ **Error Handling** - Graceful error handling and fallbacks
- 🧪 **Mock Mode** - Works without AWS for testing

### AWS Integration
- ☁️ **AWS SageMaker** - Model training and deployment
- 📦 **S3 Integration** - Data and model storage
- 🔐 **IAM Support** - Secure credential management
- 🔄 **Automatic Fallback** - Falls back to mock predictions if AWS unavailable

### Machine Learning
- 🤖 **Random Forest Classifier** - ML model for price prediction
- 📊 **Training Scripts** - Local and SageMaker training support
- 🎯 **20 Features** - Comprehensive smartphone specifications

### Documentation
- 📖 **README.md** - Complete project documentation
- 🚀 **QUICKSTART.md** - Quick start guide
- 🔧 **AWS_SETUP_GUIDE.md** - Detailed AWS setup instructions
- 📝 **This file** - Project summary

## 📂 Project Structure

```
mobile-price-predictor/
│
├── app.py                    # Flask application (main server)
├── aws_integration.py        # AWS SageMaker integration module
├── train_model.py            # Local training script
├── sagemaker_train.py        # SageMaker training script
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── env_example.txt          # Environment variables template
│
├── templates/
│   └── index.html           # Frontend HTML template
│
├── static/
│   ├── styles.css           # CSS styles
│   └── app.js               # JavaScript for frontend
│
└── docs/
    ├── README.md            # Main documentation
    ├── QUICKSTART.md        # Quick start guide
    ├── AWS_SETUP_GUIDE.md   # AWS setup guide
    └── PROJECT_SUMMARY.md   # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Application (Mock Mode - No AWS needed)
```bash
python app.py
```

### 3. Open Browser
Visit `http://localhost:5000`

### 4. Test Prediction
Fill in the form and click "Predict Price Range"

## 🔑 Key Features

### 1. **Mock Mode** (Testing without AWS)
- Works immediately without AWS setup
- Rule-based predictions for testing
- Perfect for development and demos

### 2. **AWS SageMaker Integration** (Production)
- Real ML model predictions
- Scalable cloud deployment
- Automatic fallback to mock if AWS unavailable

### 3. **Modern UI**
- Beautiful gradient design
- Responsive layout
- Smooth animations
- Form validation

### 4. **API Endpoints**
- `POST /predict` - Get price prediction
- `GET /health` - Health check
- `GET /endpoint-status` - Check AWS endpoint status

## 🎨 Price Categories

The model predicts 4 price ranges:

1. **💰 Budget** (0) - ₹0 - ₹10,000
2. **📱 Lower Mid-range** (1) - ₹10,000 - ₹20,000
3. **🎯 Upper Mid-range** (2) - ₹20,000 - ₹35,000
4. **💎 Premium** (3) - ₹35,000+

## 📊 Input Features (20 Features)

1. Battery Power (mAh)
2. Bluetooth
3. Clock Speed (GHz)
4. Dual SIM
5. Front Camera (MP)
6. 4G Support
7. Internal Memory (GB)
8. Mobile Depth (cm)
9. Mobile Weight (grams)
10. Number of Cores
11. Primary Camera (MP)
12. Pixel Height
13. Pixel Width
14. RAM (MB)
15. Screen Height (cm)
16. Screen Width (cm)
17. Talk Time (hours)
18. 3G Support
19. Touch Screen
20. WiFi

## 🔧 Configuration

### Environment Variables

Create a `.env` file from `env_example.txt`:

```env
# AWS Configuration (optional for mock mode)
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_REGION=us-east-1
SAGEMAKER_ENDPOINT=mobile-price-predictor-endpoint
SAGEMAKER_ROLE=arn:aws:iam::account:role/SageMakerRole

# Flask Configuration
FLASK_ENV=development
SECRET_KEY=your-secret-key
PORT=5000

# S3 Configuration
S3_BUCKET_NAME=your-bucket-name
```

## 📚 Documentation Files

1. **README.md** - Complete project documentation
2. **QUICKSTART.md** - Quick start guide (5 minutes)
3. **AWS_SETUP_GUIDE.md** - Step-by-step AWS setup
4. **PROJECT_SUMMARY.md** - This file

## 🧪 Testing

### Test Without AWS (Mock Mode)
```bash
python app.py
# App automatically uses mock predictions
```


## FAQs (What/Why/Alternatives)

### Problem framing
- **Task type**: Multiclass classification of price range (labels 0–3). Not regression because the dataset provides discrete price buckets, not continuous prices.
- **Metric we watch**: Overall accuracy and classwise precision/recall/F1; confusion matrix to see which ranges confuse the model.

### Model choice
- **Chosen model**: `RandomForestClassifier`.
  - **Why**: Handles nonlinear interactions, robust to outliers, minimal feature scaling, works well on tabular numeric + binary features, fast to train, good baseline with interpretability (feature importances).
  - **Tuning**: Optional `RandomizedSearchCV` with 5‑fold CV; class_weight='balanced_subsample' to reduce class imbalance bias.
  - **Result**: ~0.91 test accuracy after feature engineering.
- **Alternatives and trade‑offs**:
  - **XGBoost/LightGBM/CatBoost**: Often stronger on tabular data; more knobs, sometimes higher accuracy with longer tuning. We can swap later for more performance.
  - **Logistic Regression/Linear SVM**: Fast and simple; may underfit due to nonlinear relationships. Requires careful scaling/regularization.
  - **Neural nets**: Usually overkill for small tabular datasets; more sensitive to preprocessing.
  - **kNN**: Simple but slow at inference, sensitive to scaling, and memory heavy.

### Features used (raw + engineered)
- **Base (20) features from the dataset**: `battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory, m_dep, mobile_wt, n_cores, pc, px_height, px_width, ram, sc_h, sc_w, talk_time, three_g, touch_screen, wifi`.
- **Engineered features (why they help)**:
  - `pixel_area = px_height * px_width`: proxy for total display resolution.
  - `ppi ≈ diag(px)/sc_h`: pixel density approximation, correlates with screen quality.
  - `screen_ratio = sc_h/sc_w`: shape aspect (content/UX proxy).
  - `ram_per_core = ram/n_cores`: balanced compute/memory capacity proxy.
  - `battery_per_weight = battery_power/mobile_wt`: usability/efficiency proxy.
- These capture interactions nonlinearly and gave a measurable lift in accuracy.

### Data handling
- **Train/validation split**: Stratified 80/20 to preserve class balance.
- **No scaling/encoding required**: All inputs are numeric or binary; trees are scale‑invariant.
- **Target**: `price_range` discovered/validated in preprocessing with fallbacks for alternate names.

### Artifacts
- We save a bundle `{ model, feature_columns }` in `model.pkl` so inference code always knows the exact feature order.

### AWS components (roles and significance)
- **Amazon S3**: Durable object storage for dataset (`training/train.csv`), source code bundle (`code/code.zip`), and training outputs (`output/model.tar.gz`). Decouples data/code from compute and is accessible to both your laptop and SageMaker.
- **AWS IAM**: Secure permissions. The execution role needs S3 read/write and ECR read permissions to pull the framework container.
- **Amazon SageMaker – Training**: Managed compute to run `sagemaker_train.py` in the scikit‑learn container. Benefits: reproducible environments, logs/metrics, larger instances on demand, output model persisted to S3.
- **Amazon SageMaker – Inference (optional)**: Host the trained artifact as a HTTPS endpoint for your website/app. SageMaker handles autoscaling, health checks, and monitoring.

### Why SageMaker vs local
- **Local**: Fast iteration, zero cloud cost, great for development. We used it to prove accuracy and export a model.
- **SageMaker**: Needed when data/compute/availability requirements exceed a laptop, or when you want a managed API endpoint with monitoring and scaling.

### Deployment options
- **Simple (local)**: Flask app loads `model.pkl`, exposes `/predict`, serves HTML form.
- **Managed (cloud)**: Deploy on SageMaker endpoint; front end calls the endpoint via `boto3` or API Gateway. Alternative: Lambda + Docker for low‑cost bursty traffic.

### Cost/operational notes
- Training jobs are pay‑per‑use; endpoints are billed while running—shut down when not needed. Use `t3`/`m5` families without GPUs for this tabular model.

### Common pitfalls and how we addressed them
- Wrong feature order at inference → store and reuse `feature_columns` with the model.
- Missing permissions pulling the container → add `AmazonEC2ContainerRegistryReadOnly` to the execution role.
- S3 path mistakes (folder vs object) → always end training prefixes with `/` and name files explicitly (`train.csv`).

### How to justify choices succinctly (interview‑style)
- We framed the task as multiclass classification because labels are discrete. Chose Random Forest for strong tabular baselines with minimal preprocessing and added targeted feature engineering that captures resolution, density, aspect ratio, compute/memory balance, and battery efficiency—these correlate with price tiers. We validated with stratified splits and improved accuracy to ~0.91. We use S3 for durable data/model storage and SageMaker to run the same script in a managed container with reproducible environments and optional production deployment as an endpoint.

