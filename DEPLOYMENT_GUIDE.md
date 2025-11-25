# No-Ball Detection App - Deployment Guide

## 📋 Quick Checklist

Before deploying, ensure you have:
- [ ] `Noball_app.py` (Streamlit application)
- [ ] `noball_detection_model.keras` (trained model file)
- [ ] `requirements.txt` (dependencies)
- [ ] GitHub account (for Streamlit Cloud deployment)

---

## 🚀 Option 1: Deploy on Streamlit Cloud (RECOMMENDED)

**Advantages:**
- Free hosting
- Automatic updates from GitHub
- Easy to manage
- No credit card needed

### Steps:

1. **Create a GitHub Repository**
   - Go to https://github.com/new
   - Create a new repository named `no-ball-detection` (or similar)
   - Clone it to your local machine

2. **Add Your Files**
   ```
   your-repo/
   ├── Noball_app.py
   ├── noball_detection_model.keras
   ├── requirements.txt
   └── README.md (optional)
   ```

3. **Push to GitHub**
   
   **First time setup (one-time only):**
   ```bash
   cd path/to/your/repo
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```
   
   **Add, commit, and push your files:**
   ```bash
   git add .
   git commit -m "Initial commit: No-Ball Detection app"
   git push origin main
   ```

4. **Deploy on Streamlit Cloud**
   - Go to https://streamlit.io/cloud
   - Click "New app"
   - Select your GitHub repo, branch, and main file (`Noball_app.py`)
   - Click "Deploy"
   - Wait 2-3 minutes for deployment to complete

5. **Share Your App**
   - Your app URL will be: `https://[your-username]-[repo-name].streamlit.app`

---

## 📤 Detailed Guide: How to Push to GitHub

### Prerequisites:
- Git installed on your computer ([download here](https://git-scm.com/downloads))
- GitHub account ([create one here](https://github.com/join))
- Your files ready (`Noball_app_quantized.py`, `noball_detection_model_quantized.tflite`, `requirements.txt`)

### Step-by-Step Instructions:

#### **Step 1: Create a GitHub Repository**
1. Go to https://github.com/new
2. Enter repository name: `no-ball-detection`
3. Add description: "Cricket No-Ball Detection ML App"
4. Choose "Public" (so others can see it)
5. **DO NOT** check "Initialize with README"
6. Click "Create repository"

#### **Step 2: Open Command Prompt/Terminal**

**Windows:**
- Press `Win + R`, type `cmd`, press Enter
- Or open PowerShell

**Mac/Linux:**
- Open Terminal

#### **Step 3: Navigate to Your Project Folder**
```bash
cd C:\Users\YourUsername\Downloads\SportsEng
```
Replace `YourUsername` with your actual Windows username.

#### **Step 4: Initialize Git (First Time Only)**
```bash
git init
git config --global user.name "Your GitHub Username"
git config --global user.email "your.email@gmail.com"
```

#### **Step 5: Add Remote Repository**
Copy this from your GitHub repo page (after you created it):
```bash
git remote add origin https://github.com/YOUR_USERNAME/no-ball-detection.git
```
Replace `YOUR_USERNAME` with your GitHub username.

#### **Step 6: Add Your Files**
```bash
git add .
```
This stages all files in your folder.

#### **Step 7: Commit Your Changes**
```bash
git commit -m "Initial commit: Add No-Ball Detection app with quantized model"
```

#### **Step 8: Push to GitHub**
```bash
git branch -M main
git push -u origin main
```

**First time pushing?** It will ask for:
- GitHub username
- GitHub password (use Personal Access Token instead - see below)

#### **Using Personal Access Token (Recommended)**

If you get authentication errors:

1. Go to https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Give it a name: `git-push`
4. Check: `repo` (full control of private repositories)
5. Click "Generate token"
6. **Copy the token** (you won't see it again!)
7. When Git asks for password, **paste the token**

### Common Issues & Solutions:

| Problem | Solution |
|---------|----------|
| "fatal: not a git repository" | Run `git init` first |
| "fatal: 'origin' does not appear to be a 'git' repository" | Check your remote: `git remote -v` |
| "fatal: unable to access... 401 Unauthorized" | Use Personal Access Token instead of password |
| "Please tell me who you are" | Run `git config --global user.name "YourName"` |
| "nothing to commit" | Make sure you have files in the folder |

### Verify Success:
```bash
git log
```
You should see your commit message. Refresh your GitHub repo page to see the files online!

---

## 🟢 Option 2: Deploy on Render (FREE Tier Available)

**Advantages:**
- Free tier available
- Auto-deploys from GitHub
- Reliable uptime

### Steps:

1. **Push your code to GitHub** (same as Option 1, steps 1-3)

2. **Create Render Account**
   - Go to https://render.com
   - Sign in with GitHub

3. **Deploy New Web Service**
   - Click "New" → "Web Service"
   - Connect your GitHub repo
   - Set build command: `pip install -r requirements.txt`
   - Set start command: `streamlit run Noball_app.py`
   - Choose Free plan
   - Deploy

---

## 🔵 Option 3: Deploy on Railway.app

**Advantages:**
- Simple deployment
- Good free tier

### Steps:

1. **Sign up at https://railway.app**
   - Connect your GitHub account

2. **Create New Project**
   - Click "Deploy from GitHub repo"
   - Select your repository

3. **Add Environment Variables** (if needed)
   - Railway auto-detects Python apps

4. **Deploy**
   - Railway automatically builds and deploys

---

## 💻 Option 4: Run Locally

Perfect for testing before deploying to the cloud:

```bash
# Navigate to your app directory
cd path/to/your/app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run Noball_app.py
```

Your app will be available at: `http://localhost:8501`

---

## 📦 About Model Size

The trained model file (`noball_detection_model.keras`) may be large. 

**If your model is > 100MB:**
- Use Git LFS (Large File Storage) for GitHub
- Or upload model to cloud storage (Google Drive, AWS S3) and load it in your app

**To use Git LFS:**
```bash
git lfs install
git lfs track "*.keras"
git add .gitattributes
git push
```

---

## 🔧 Troubleshooting

### "Module not found" errors
- Ensure all imports in `Noball_app.py` are in `requirements.txt`
- Update `requirements.txt` if needed

### Model file not found
- Ensure `noball_detection_model.keras` is in the same directory as `Noball_app.py`
- Check file path in the app code

### App crashes on upload
- Model might be loading on every run
- Use `@st.cache_resource` decorator (already in your app)

### Memory limit exceeded
- Streamlit Cloud has 1GB limit
- Consider model optimization or quantization

---

## 📝 Example README.md

```markdown
# No-Ball Cricket Detection App

A machine learning app to detect whether a cricket bowl is legal or a no-ball.

## How to Use

1. Upload a cricket delivery image
2. The app predicts if it's "Legal" or "No ball"
3. View confidence scores and results

## Local Deployment

```bash
pip install -r requirements.txt
streamlit run Noball_app.py
```

## Cloud Deployment

Deployed on Streamlit Cloud at: [YOUR_APP_URL]

## Model

- Input Size: 128x128 RGB images
- Architecture: CNN with 3 Conv2D layers
- Classes: Legal, No ball
- Accuracy: [YOUR_ACCURACY]%
```

---

## ✅ Deployment Checklist

- [ ] Model saved as `.keras` format
- [ ] `requirements.txt` includes all dependencies
- [ ] `Noball_app.py` paths are relative
- [ ] GitHub repo created and files pushed
- [ ] Deployment platform connected
- [ ] App tested locally first
- [ ] App URL shared with team
- [ ] Model performance documented

---

## 🎉 After Deployment

1. **Test your live app** - Upload test images
2. **Monitor logs** - Check for errors on platform dashboard
3. **Share the URL** - Send link to your team
4. **Collect feedback** - Iterate and improve

---

Need help? Check:
- Streamlit Docs: https://docs.streamlit.io
- Render Docs: https://render.com/docs
- Railway Docs: https://docs.railway.app
