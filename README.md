# 🏏 No-Ball Cricket Detection App

A machine learning-powered web application to detect whether a cricket delivery is **Legal** or a **No ball**.

## 🚀 Live App

**Access the app here:** https://noballdetection-nekjmesettduyrwfpma2ly.streamlit.app

## 📋 Features

- ✅ Upload cricket delivery images (JPG, JPEG, PNG)
- ✅ Real-time predictions using TensorFlow Lite
- ✅ Confidence scores for each prediction
- ✅ Probability breakdown for both classes
- ✅ Fast inference with quantized model (~8-10 MB)
- ✅ Demo mode fallback if model unavailable

## 🛠️ Local Setup

### Requirements
```bash
pip install -r requirements.txt
```

### Run Locally
```bash
streamlit run Noball_app_quantized.py
```

Your app will be available at: `http://localhost:8501`

## 📊 Model Info

| Aspect | Details |
|--------|---------|
| **Input Size** | 128×128 RGB Images |
| **Model Type** | Quantized CNN (TensorFlow Lite) |
| **File Size** | ~8-10 MB |
| **Classes** | Legal, No ball |
| **Framework** | TensorFlow 2.x |
| **Quantization** | INT8 |

## 📁 File Structure

```
no-ball-detection/
├── Noball_app_quantized.py           # Main Streamlit app
├── noball_detection_model_quantized.tflite   # Pre-trained model
├── requirements.txt                   # Python dependencies
├── DEPLOYMENT_GUIDE.md               # Detailed deployment instructions
└── README.md                         # This file
```

## 🔍 How to Use

1. **Visit the app:** https://noballdetection-nekjmesettduyrwfpma2ly.streamlit.app
2. **Upload an image** of a cricket delivery (JPG/PNG)
3. **Wait for analysis** - typically < 2 seconds
4. **View results:**
   - ✅ Legal delivery (with confidence %)
   - 🚫 No ball delivery (with confidence %)
5. **Expand "Probability Breakdown"** to see all class probabilities

## ⚙️ Technologies Used

- **Framework:** Streamlit
- **ML Framework:** TensorFlow / TensorFlow Lite
- **Image Processing:** PIL (Pillow)
- **Numerical Computing:** NumPy
- **Hosting:** Streamlit Cloud

## 🐛 Troubleshooting

### App shows "error occurred"
- **Solution:** Refresh the page (Ctrl+F5)
- Check browser console for errors
- Try a different image format

### Predictions not working
- **Status:** App may be in demo mode if TensorFlow unavailable
- Demo mode shows simulated predictions (65% Legal, 35% No ball)
- Actual model predictions work once TensorFlow loads

### Image upload fails
- **Supported formats:** JPG, JPEG, PNG only
- **Max size:** ~5 MB recommended
- **Resolution:** Works best with 128×128 or larger images

### Slow predictions
- First prediction may take longer (model loading)
- Subsequent predictions are cached and faster
- App runs on Streamlit Cloud free tier (standard performance)

## 📈 Performance

- **Model Size:** 8-10 MB (quantized)
- **Inference Time:** ~200-500 ms per image
- **Accuracy:** Depends on training data quality
- **Deployment:** Streamlit Cloud (always free tier)

## 🔄 Deployment

### Deployed on Streamlit Cloud
- **Repository:** https://github.com/InfernousFlame/No_Ball_detection
- **Branch:** main
- **File:** Noball_app_quantized.py
- **Auto-deploy:** Enabled (updates on every GitHub push)

### To redeploy:
1. Push changes to GitHub
2. Streamlit Cloud auto-detects and redeploys
3. Takes 2-3 minutes to complete

## 📝 Training the Model

If you want to train your own model:

1. Use the Jupyter notebook: `Bowler_Extraction_&_No_Ball_Check.ipynb`
2. Prepare your cricket delivery images dataset
3. Run the training cells
4. Export model as `.tflite` (TensorFlow Lite format)
5. Replace `noball_detection_model_quantized.tflite`
6. Push to GitHub for auto-deployment

## 🤝 Contributing

To improve the model:
1. Collect more cricket delivery images
2. Retrain the model
3. Quantize to TensorFlow Lite
4. Update the model file
5. Push to GitHub

## 📄 License

This project is open source. Feel free to use and modify!

## 💬 Support

For issues or questions:
- Check the DEPLOYMENT_GUIDE.md for detailed troubleshooting
- Review the code comments in Noball_app_quantized.py
- Check Streamlit Cloud dashboard logs

## 🎯 Future Enhancements

- [ ] Add batch image processing
- [ ] Implement model explanations (LIME/SHAP)
- [ ] Add webcam/video support
- [ ] Create mobile app
- [ ] Deploy on additional platforms (AWS, GCP, Azure)
- [ ] Improve model accuracy with more training data

---

**Created with ❤️ for cricket enthusiasts and ML engineers**

🏏 **Happy Detecting!**
