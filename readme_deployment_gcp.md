# Deploy the agent to GCP Cloud run


# ✅ STEP 1 — Build your Docker image locally

From the root of your project (where the Dockerfile is):

```bash
docker build -t ai-newsletter .
```

You now have a local image named `ai-newsletter`. You can test it with 
```bash
docker run -p 8080:8080 ai-newsletter
```

---

# ✅ STEP 2 — Configure gcloud

```bash
gcloud config set project ai-newslteer
gcloud config set run/region us-central1
```

---

# ✅ STEP 3 — Enable required APIs

```bash
gcloud services enable \
    artifactregistry.googleapis.com \
    run.googleapis.com \
    cloudbuild.googleapis.com
```

---

# ✅ STEP 4 — Create Artifact Registry repo (Docker type)

```bash
gcloud artifacts repositories create ai-newsletter-repo \
  --repository-format=docker \
  --location=us-central1 \
  --description="Newsletter agent container repo"
```

If it already exists, you will get a harmless error — you can ignore it.

---

# ✅ STEP 5 — Tag the local Docker image for Artifact Registry

```bash
docker tag ai-newsletter \
  us-central1-docker.pkg.dev/ai-newslteer/ai-newsletter-repo/ai-newsletter:latest
```

---

# ✅ STEP 6 — Authenticate Docker to push to Artifact Registry

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
```

---

# ✅ STEP 7 — Push the image to Artifact Registry

```bash
docker push \
  us-central1-docker.pkg.dev/ai-newslteer/ai-newsletter-repo/ai-newsletter:latest
```

---

# ✅ STEP 8 — Deploy to Cloud Run

Use the pushed image:

```bash
gcloud run deploy ai-newsletter-service \
  --image us-central1-docker.pkg.dev/ai-newslteer/ai-newsletter-repo/ai-newsletter:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars PORT=8080
```

---

# 🎉 DEPLOYMENT COMPLETE

You can get the service URL:

```bash
gcloud run services describe ai-newsletter-service \
  --format="value(status.url)" \
  --region us-central1
```

Open the printed URL in your browser — you’ll see the **ADK Web UI**.

---


# How to deal with the Environment Variables
---


# 🔐 **1. Use Secret Manager for Sensitive Fields**

### Step A — Create secrets:

```bash
echo -n "YOUR_API_KEY" | gcloud secrets create google-api-key --data-file=-
echo -n "YOUR_SMTP_PASSWORD" | gcloud secrets create smtp-pass --data-file=-
```
(Repeat for other variables)
### Step B — Grant Cloud Run access:

```bash
gcloud secrets add-iam-policy-binding google-api-key \
  --member=serviceAccount:$(gcloud projects describe ai-newslteer --format='value(projectNumber)')-compute@developer.gserviceaccount.com \
  --role=roles/secretmanager.secretAccessor
```

(Repeat for smtp-pass)

### Step C — Deploy with secrets:

```bash
gcloud run deploy ai-newsletter \
  --image gcr.io/ai-newslteer/ai-newsletter \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_GENAI_USE_VERTEXAI=FALSE \
  --set-env-vars SMTP_HOST=smtp.gmail.com \
  --set-env-vars SMTP_PORT=587 \
  --set-secrets SMTP_USER=smtp_user:latest \
  --set-secrets NEWSLETTER_FROM_EMAIL=from_email:latest \
  --set-secrets GOOGLE_API_KEY=google-api-key:latest \
  --set-secrets SMTP_PASS=smtp-pass:latest
```

---

# 🎯 **2. Verify Environment Variables on Cloud Run**

After deployment:

* Go to **Cloud Run → ai-newsletter → Revisions → Environment Variables**
* You should see all of them set.

---
