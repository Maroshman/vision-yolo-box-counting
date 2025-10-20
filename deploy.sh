#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="rm-ai-api"
REGION="us-central1"
SERVICE_NAME="yolo-box-counting-api"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo -e "${BLUE}🚀 YOLO Box Counting API - Google Cloud Deployment${NC}"
echo "=================================================="

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI is not installed. Please install it first.${NC}"
    echo "https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Get or set project ID
if [ -z "$PROJECT_ID" ]; then
    echo -e "${YELLOW}📝 No PROJECT_ID set. Getting current project...${NC}"
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
    
    if [ -z "$PROJECT_ID" ]; then
        echo -e "${RED}❌ No Google Cloud project set. Please run:${NC}"
        echo "gcloud config set project YOUR_PROJECT_ID"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Using project: ${PROJECT_ID}${NC}"
fi

# Update PROJECT_ID in the script and service file
sed -i.bak "s/PROJECT_ID=\"\"/PROJECT_ID=\"${PROJECT_ID}\"/" deploy.sh
sed -i.bak "s/PROJECT_ID/${PROJECT_ID}/g" cloud-run-service.yaml

echo -e "${BLUE}🔧 Setting up Google Cloud services...${NC}"

# Enable required APIs
echo "Enabling Cloud Run API..."
gcloud services enable run.googleapis.com --project=${PROJECT_ID}

echo "Enabling Container Registry API..."
gcloud services enable containerregistry.googleapis.com --project=${PROJECT_ID}

echo "Enabling Cloud Build API..."
gcloud services enable cloudbuild.googleapis.com --project=${PROJECT_ID}

# Configure Docker for gcloud
echo -e "${BLUE}🐳 Configuring Docker authentication...${NC}"
gcloud auth configure-docker --quiet

# Build the Docker image
echo -e "${BLUE}🏗️ Building Docker image...${NC}"
docker build -f Dockerfile.api -t ${IMAGE_NAME}:latest .

# Push the image to Google Container Registry
echo -e "${BLUE}📤 Pushing image to Google Container Registry...${NC}"
docker push ${IMAGE_NAME}:latest

# Create secrets for API keys (optional - you can set these manually)
echo -e "${BLUE}🔐 Setting up secrets (optional)...${NC}"
echo "You can create secrets for your API keys:"
echo "gcloud secrets create api-secrets --data-file=secrets.json"
echo ""

# Deploy to Cloud Run
echo -e "${BLUE}🚀 Deploying to Cloud Run...${NC}"
gcloud run deploy ${SERVICE_NAME} \
    --image=${IMAGE_NAME}:latest \
    --platform=managed \
    --region=${REGION} \
    --allow-unauthenticated \
    --port=8080 \
    --cpu=2 \
    --memory=4Gi \
    --max-instances=10 \
    --timeout=300 \
    --set-env-vars="DETECTION_BACKEND=roboflow,ENABLE_OCR=true" \
    --project=${PROJECT_ID}

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format='value(status.url)' --project=${PROJECT_ID})

echo ""
echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
echo "=================================================="
echo -e "${GREEN}✅ Service URL: ${SERVICE_URL}${NC}"
echo -e "${GREEN}✅ API Docs: ${SERVICE_URL}/docs${NC}"
echo -e "${GREEN}✅ Health Check: ${SERVICE_URL}/health${NC}"
echo ""
echo -e "${YELLOW}📝 Next steps:${NC}"
echo "1. Set up your API keys as environment variables in Cloud Run console"
echo "2. Test the API: curl ${SERVICE_URL}/health"
echo "3. Monitor logs: gcloud logs tail /projects/${PROJECT_ID}/logs/run.googleapis.com%2Fstdout"
echo ""
echo -e "${BLUE}🔧 Useful commands:${NC}"
echo "View logs: gcloud logs tail /projects/${PROJECT_ID}/logs/run.googleapis.com%2Fstdout"
echo "Update service: gcloud run services replace cloud-run-service.yaml --region=${REGION}"
echo "Delete service: gcloud run services delete ${SERVICE_NAME} --region=${REGION}"