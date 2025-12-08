# 🚀 Deploy Python Backend to Render (FREE)

Render offers free hosting for Python apps with a permanent URL. This is better than ngrok for production.

## 📋 Prerequisites

1. GitHub account
2. Render account (free): https://render.com
3. Your code in a GitHub repository

---

## Step 1: Prepare Your Project

### 1.1 Create requirements.txt

Make sure you have all dependencies listed:

```txt
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6
opencv-python-headless>=4.8.0
numpy>=1.24.0
rasterio>=1.3.0
Pillow>=10.0.0
torch>=2.0.0
torchvision>=0.15.0
```

**Important**: Use `opencv-python-headless` instead of `opencv-python` for server deployment (no GUI dependencies).

### 1.2 Update requirements.txt in your project

Run this command to update:

```bash
pip freeze > requirements.txt
```

Then edit it to ensure it has the packages above.

### 1.3 Create render.yaml (Optional but Recommended)

Create a file called `render.yaml` in your project root:

```yaml
services:
  - type: web
    name: building-analysis-api
    env: python
    region: singapore
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn backend_api:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.12.0
```

---

## Step 2: Push to GitHub

### 2.1 Initialize Git (if not already done)

```bash
git init
git add .
git commit -m "Initial commit for Render deployment"
```

### 2.2 Create GitHub Repository

1. Go to https://github.com/new
2. Create a new repository (e.g., "building-analysis-backend")
3. Don't initialize with README (you already have code)

### 2.3 Push Your Code

```bash
git remote add origin https://github.com/YOUR_USERNAME/building-analysis-backend.git
git branch -M main
git push -u origin main
```

**Important Files to Include:**
- `backend_api.py`
- `rooftop_change_detection.py`
- `synthetic_parcel_analyzer.py`
- `rooftop_model_loader.py`
- `rooftop_preprocessing.py`
- `rooftop_best_model_new.pt` (your trained model)
- `requirements.txt`
- `render.yaml` (optional)

**Files to Exclude** (add to .gitignore):
- `__pycache__/`
- `*.pyc`
- `.env`
- `venv/`
- `node_modules/`
- Test images (too large)

---

## Step 3: Deploy on Render

### 3.1 Sign Up / Log In

Go to https://render.com and sign up with GitHub.

### 3.2 Create New Web Service

1. Click **"New +"** button
2. Select **"Web Service"**
3. Connect your GitHub repository
4. Select your repository from the list

### 3.3 Configure the Service

Fill in these settings:

**Basic Settings:**
- **Name**: `building-analysis-api` (or your choice)
- **Region**: Choose closest to you (e.g., Singapore, Oregon)
- **Branch**: `main`
- **Root Directory**: Leave blank (unless your code is in a subfolder)

**Build & Deploy:**
- **Runtime**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn backend_api:app --host 0.0.0.0 --port $PORT`

**Plan:**
- Select **"Free"** plan

### 3.4 Environment Variables (Optional)

If you need any environment variables, add them in the "Environment" section.

### 3.5 Deploy!

Click **"Create Web Service"**

Render will:
1. Clone your repository
2. Install dependencies (takes 5-10 minutes first time)
3. Start your app
4. Give you a URL like: `https://building-analysis-api.onrender.com`

---

## Step 4: Update Frontend

Once deployed, you'll get a URL like:
```
https://building-analysis-api.onrender.com
```

### 4.1 Update Frontend .env

Edit `project/.env`:

```env
VITE_API_URL=https://building-analysis-api.onrender.com
```

### 4.2 Rebuild Frontend

```bash
cd project
npm run build
```

### 4.3 Upload to Netlify

Drag the new `project/dist` folder to Netlify.

---

## Step 5: Test Your Deployment

### 5.1 Check API Health

Visit: `https://your-app.onrender.com/`

Should see:
```json
{"status":"ok","message":"Building Analysis API is running"}
```

### 5.2 Test from Netlify

Go to your Netlify site and try uploading images!

---

## 🎯 Important Notes

### Free Tier Limitations

**Render Free Tier:**
- ✅ Permanent URL (doesn't change)
- ✅ Automatic HTTPS
- ✅ 750 hours/month (enough for demos)
- ⚠️ **Cold starts**: App sleeps after 15 min of inactivity
- ⚠️ First request after sleep takes 30-60 seconds
- ⚠️ 512 MB RAM limit

### Cold Start Solution

The first request after inactivity will be slow. You can:
1. Use a service like UptimeRobot to ping your API every 10 minutes
2. Or just accept the cold start (fine for demos)

### Model File Size

Your `rooftop_best_model_new.pt` file might be large. If GitHub rejects it:

**Option A: Use Git LFS**
```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add rooftop_best_model_new.pt
git commit -m "Add model with LFS"
git push
```

**Option B: Host model elsewhere**
- Upload to Google Drive or Dropbox
- Download in your code on startup
- Update `rooftop_model_loader.py` to download if not exists

---

## 🔧 Troubleshooting

### Build Fails

**Error: "Could not find a version that satisfies the requirement torch"**

Solution: Torch is large. Add to `requirements.txt`:
```txt
--extra-index-url https://download.pytorch.org/whl/cpu
torch>=2.0.0
torchvision>=0.15.0
```

**Error: "opencv-python requires GUI"**

Solution: Use `opencv-python-headless` instead.

### App Crashes

Check logs in Render dashboard:
1. Go to your service
2. Click "Logs" tab
3. Look for Python errors

### Out of Memory

Free tier has 512 MB RAM. If your model is too large:
- Upgrade to paid plan ($7/month for 2GB RAM)
- Or optimize your model

### Slow Performance

Free tier has limited CPU. For faster inference:
- Upgrade to paid plan
- Or use smaller images
- Or reduce batch size in patch processing

---

## 🔄 Updating Your App

### Auto-Deploy (Recommended)

Render auto-deploys when you push to GitHub:

```bash
# Make changes to your code
git add .
git commit -m "Fix bug"
git push
```

Render will automatically rebuild and redeploy!

### Manual Deploy

In Render dashboard:
1. Go to your service
2. Click "Manual Deploy"
3. Select branch
4. Click "Deploy"

---

## 💰 Cost Comparison

| Service | Free Tier | Pros | Cons |
|---------|-----------|------|------|
| **ngrok** | ✅ Yes | Fast, local | URL changes, must keep running |
| **Render** | ✅ Yes | Permanent URL, auto-deploy | Cold starts, 512MB RAM |
| **Railway** | ✅ $5 credit | Fast, no cold starts | Credit runs out |
| **Heroku** | ❌ No | Reliable | $7/month minimum |
| **AWS/GCP** | ⚠️ Complex | Scalable | Requires setup, can be expensive |

**Recommendation**: Use Render for free permanent hosting!

---

## 📝 Quick Checklist

Before deploying to Render:

- [ ] `requirements.txt` has all dependencies
- [ ] Using `opencv-python-headless` not `opencv-python`
- [ ] All Python files are in repository
- [ ] Model file (`rooftop_best_model_new.pt`) is included
- [ ] Code pushed to GitHub
- [ ] `.gitignore` excludes unnecessary files
- [ ] Tested locally with `uvicorn backend_api:app --host 0.0.0.0 --port 8000`

After deploying:

- [ ] Check logs for errors
- [ ] Test health endpoint: `https://your-app.onrender.com/`
- [ ] Update frontend `.env` with new URL
- [ ] Rebuild frontend: `npm run build`
- [ ] Upload new `dist` to Netlify
- [ ] Test full workflow on Netlify site

---

## 🎉 You're Done!

Your app is now:
- ✅ Backend on Render (permanent, free)
- ✅ Frontend on Netlify (permanent, free)
- ✅ Fully functional and shareable!

Share your Netlify URL with anyone and it will work!

---

## 🆘 Need Help?

Common issues:

1. **"Module not found"**: Add to `requirements.txt`
2. **"Port already in use"**: Render handles this automatically
3. **"Out of memory"**: Model too large, upgrade plan
4. **"Cold start too slow"**: Use UptimeRobot to keep warm

Check Render docs: https://render.com/docs
