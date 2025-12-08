# ⚡ Quick Render Setup (5 Minutes)

## Step 1: Push to GitHub

```bash
# Initialize git (if not done)
git init

# Add files
git add backend_api.py rooftop_change_detection.py synthetic_parcel_analyzer.py
git add rooftop_model_loader.py rooftop_preprocessing.py
git add rooftop_best_model_new.pt
git add requirements.txt render.yaml .gitignore

# Commit
git commit -m "Deploy to Render"

# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

## Step 2: Deploy on Render

1. Go to https://render.com
2. Sign up with GitHub
3. Click **"New +"** → **"Web Service"**
4. Connect your repository
5. Settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn backend_api:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free
6. Click **"Create Web Service"**
7. Wait 5-10 minutes for build

## Step 3: Get Your URL

You'll get: `https://your-app-name.onrender.com`

## Step 4: Update Frontend

```bash
# Edit project/.env
VITE_API_URL=https://your-app-name.onrender.com

# Rebuild
cd project
npm run build

# Upload dist/ to Netlify
```

## Done! 🎉

Your app is now live with a permanent URL!

---

## ⚠️ Important Notes

1. **First request is slow** (30-60 sec) - this is normal for free tier
2. **Model file**: If GitHub rejects large files, use Git LFS:
   ```bash
   git lfs install
   git lfs track "*.pt"
   git add .gitattributes
   git commit -m "Add LFS"
   git push
   ```

3. **Check logs** in Render dashboard if something fails

---

## 🔄 To Update Later

Just push to GitHub:
```bash
git add .
git commit -m "Update"
git push
```

Render auto-deploys! ✨
