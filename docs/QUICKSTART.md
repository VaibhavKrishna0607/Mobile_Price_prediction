# ⚡ Quick Start Guide

Get the Mobile Price Predictor up and running in 5 minutes!

## Option 1: Run Without AWS (Mock Mode)

Perfect for testing and development without AWS setup.

### Steps:

1. **Install Dependencies**:
```bash
cd mobile-price-predictor
pip install -r requirements.txt
```

2. **Run the Application**:
```bash
python app.py
```

3. **Open Browser**:
   - Navigate to `http://localhost:5000`
   - The app will automatically use mock predictions (no AWS needed!)

4. **Test Prediction**:
   - Fill in the form with smartphone specifications
   - Click "Predict Price Range"
   - See the prediction result!

**That's it!** The app runs in mock mode by default when AWS credentials are not configured.

## Option 2: Run With AWS SageMaker

For production use with real ML model predictions.

### Prerequisites:
- AWS Account
- AWS credentials (Access Key ID and Secret Access Key)
- SageMaker endpoint deployed

### Steps:

1. **Set Up AWS** (first time only):
   - Follow the [AWS_SETUP_GUIDE.md](AWS_SETUP_GUIDE.md)
   - Create IAM role, S3 bucket, and deploy model

2. **Configure Environment**:
   - Copy `env_example.txt` to `.env`
   - Fill in your AWS credentials:
   ```env
   AWS_ACCESS_KEY_ID=your-key
   AWS_SECRET_ACCESS_KEY=your-secret
   AWS_REGION=us-east-1
   SAGEMAKER_ENDPOINT=mobile-price-predictor-endpoint
   ```

3. **Install and Run**:
```bash
pip install -r requirements.txt
python app.py
```

4. **Verify Connection**:
   - Visit `http://localhost:5000/health`
   - Should show `"mode": "aws"` if connected

## Testing

### Test the API:

```bash
# Health check
curl http://localhost:5000/health

# Prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "battery_power": 2000,
    "ram": 3072,
    "int_memory": 32,
    "clock_speed": 2.0,
    "n_cores": 4,
    "pc": 12,
    "fc": 5,
    "px_height": 1960,
    "px_width": 1080,
    "sc_h": 12.5,
    "sc_w": 6.2,
    "touch_screen": 1,
    "mobile_wt": 180,
    "m_dep": 0.8,
    "talk_time": 10,
    "three_g": 1,
    "four_g": 1,
    "wifi": 1,
    "blue": 1,
    "dual_sim": 1
  }'
```

## Troubleshooting

### Port Already in Use
```bash
# Use a different port
PORT=5001 python app.py
```

### Module Not Found
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### AWS Connection Issues
- Check your `.env` file has correct credentials
- Verify endpoint exists in SageMaker Console
- App will fall back to mock mode automatically

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [AWS_SETUP_GUIDE.md](AWS_SETUP_GUIDE.md) for AWS integration
- Customize the frontend in `templates/index.html` and `static/styles.css`

---

**Happy Predicting!** 🚀

