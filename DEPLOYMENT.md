# 🚀 Deployment Guide - Ethical AI Auditor SaaS

Complete guide for deploying the Ethical AI Auditor on various platforms.

## Table of Contents
1. [Streamlit Cloud Deployment](#streamlit-cloud)
2. [Heroku Deployment](#heroku)
3. [AWS Deployment](#aws)
4. [Google Cloud Platform](#gcp)
5. [Docker Deployment](#docker)
6. [Blockchain Configuration](#blockchain)

---

## 1. Streamlit Cloud Deployment (Recommended) {#streamlit-cloud}

**Best for**: Quick demos, prototypes, small teams
**Cost**: Free tier available
**Setup Time**: 5-10 minutes

### Prerequisites
- GitHub account
- Streamlit Cloud account (free at share.streamlit.io)

### Steps

#### A. Prepare Repository

1. **Create GitHub Repository**
```bash
# Initialize git
git init
git add app.py requirements.txt README.md
git commit -m "Initial commit: Ethical AI Auditor"

# Create repo on GitHub and push
git remote add origin https://github.com/AryHHARy/ethical-ai-auditor-SaaS.git
git branch -M main
git push -u origin main
```

2. **Verify Files**
Ensure your repository contains:
- ✅ `app.py` (main application)
- ✅ `requirements.txt` (dependencies)
- ✅ `README.md` (documentation)

#### B. Deploy to Streamlit Cloud

1. **Sign Up**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "Sign up with GitHub"
   - Authorize Streamlit Cloud

2. **Create New App**
   - Click "New app" button
   - Select your repository: `yourusername/ethical-ai-auditor`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: `yourapp-name` (customize as desired)

3. **Configure Settings**
   - Click "Advanced settings" (optional)
   - Python version: 3.9 or higher
   - Click "Deploy"

4. **Wait for Deployment**
   - Initial deployment: 2-5 minutes
   - Watch build logs for errors
   - App will auto-launch when ready

#### C. Configure Secrets (for Blockchain)

1. **Access App Settings**
   - In Streamlit Cloud dashboard, select your app
   - Click "Settings" (⚙️ icon)
   - Select "Secrets"

2. **Add Configuration**
```toml
# Ethereum Configuration
INFURA_URL = "https://sepolia.infura.io/v3/YOUR_INFURA_PROJECT_ID"
PRIVATE_KEY = "your_ethereum_wallet_private_key_without_0x"
CONTRACT_ADDRESS = "0xYourDeployedSmartContractAddress"

# Optional: Database
DATABASE_URL = "postgresql://user:pass@host:5432/dbname"
```

3. **Save and Restart**
   - Click "Save"
   - App will automatically restart with new secrets

#### D. Custom Domain (Optional)

1. In app settings, go to "General"
2. Add custom domain: `app.yourdomain.com`
3. Update DNS records:
   ```
   CNAME: app.yourdomain.com → yourapp.streamlit.app
   ```

### Troubleshooting

**Build Fails**
- Check `requirements.txt` for typos
- Verify Python version compatibility
- Review build logs for specific errors

**App Crashes**
- Check memory usage (free tier: 1GB limit)
- Optimize data processing for large files
- Use caching with `@st.cache_data`

**Secrets Not Working**
- Ensure proper TOML format
- No quotes around keys
- Restart app after changes

---

## 2. Heroku Deployment {#heroku}

**Best for**: Production apps, custom domains, add-ons
**Cost**: Starting at $7/month
**Setup Time**: 15-20 minutes

### Prerequisites
- Heroku account
- Heroku CLI installed

### Steps

#### A. Prepare Files

1. **Create Procfile**
```bash
echo "web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile
```

2. **Create setup.sh**
```bash
cat > setup.sh << EOF
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = \$PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
EOF
```

3. **Update requirements.txt**
```txt
streamlit==1.31.0
numpy==1.24.3
pandas==2.0.3
torch==2.1.0
matplotlib==3.7.2
web3==6.11.3
ecdsa==0.18.0
Pillow==10.0.0
```

#### B. Deploy to Heroku

```bash
# Login to Heroku
heroku login

# Create Heroku app
heroku create ethical-ai-auditor

# Set buildpack
heroku buildpacks:set heroku/python

# Deploy
git add .
git commit -m "Prepare for Heroku deployment"
git push heroku main

# Open app
heroku open
```

#### C. Configure Environment Variables

```bash
# Set blockchain config
heroku config:set INFURA_URL="https://sepolia.infura.io/v3/YOUR_KEY"
heroku config:set PRIVATE_KEY="your_private_key"
heroku config:set CONTRACT_ADDRESS="0xYourContract"
```

#### D. Scale Dynos

```bash
# Scale up (for production)
heroku ps:scale web=1:standard-1x

# View logs
heroku logs --tail
```

---

## 3. AWS Deployment {#aws}

**Best for**: Enterprise, high traffic, full control
**Cost**: Variable (EC2 starting ~$10/month)
**Setup Time**: 30-45 minutes

### Option A: EC2 with Nginx

#### Setup EC2 Instance

1. **Launch EC2**
   - AMI: Ubuntu 22.04 LTS
   - Instance Type: t2.medium (2 vCPU, 4GB RAM)
   - Security Group: Allow ports 22, 80, 443, 8501

2. **Connect to Instance**
```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

3. **Install Dependencies**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python
sudo apt install python3-pip python3-venv -y

# Install Nginx
sudo apt install nginx -y
```

4. **Setup Application**
```bash
# Clone repository
git clone https://github.com/AryHHAry/ethical-ai-auditor-SaaS.git
cd ethical-ai-auditor

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

5. **Configure Nginx**
```bash
sudo nano /etc/nginx/sites-available/streamlit
```

Add:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/streamlit /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

6. **Setup Systemd Service**
```bash
sudo nano /etc/systemd/system/streamlit.service
```

Add:
```ini
[Unit]
Description=Streamlit Ethical AI Auditor
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/ethical-ai-auditor
Environment="PATH=/home/ubuntu/ethical-ai-auditor/venv/bin"
ExecStart=/home/ubuntu/ethical-ai-auditor/venv/bin/streamlit run app.py

[Install]
WantedBy=multi-user.target
```

Start service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable streamlit
sudo systemctl start streamlit
sudo systemctl status streamlit
```

### Option B: ECS with Docker

See [Docker Deployment](#docker) section, then:

1. Push image to ECR
2. Create ECS cluster
3. Define task definition
4. Create service
5. Configure load balancer

---

## 4. Google Cloud Platform {#gcp}

**Best for**: Google ecosystem, AI/ML workloads
**Cost**: Free tier available, then ~$10-50/month
**Setup Time**: 20-30 minutes

### Option A: Cloud Run (Serverless)

1. **Build Container**
```bash
# Install gcloud CLI
gcloud init

# Build and push
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/ethical-ai-auditor

# Deploy
gcloud run deploy ethical-ai-auditor \
  --image gcr.io/YOUR_PROJECT_ID/ethical-ai-auditor \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8501
```

2. **Set Environment Variables**
```bash
gcloud run services update ethical-ai-auditor \
  --set-env-vars INFURA_URL=your_url,PRIVATE_KEY=your_key
```

### Option B: App Engine

1. **Create app.yaml**
```yaml
runtime: python39
entrypoint: streamlit run app.py --server.port=$PORT

instance_class: F2

automatic_scaling:
  min_instances: 1
  max_instances: 10

env_variables:
  INFURA_URL: "your_infura_url"
```

2. **Deploy**
```bash
gcloud app deploy
gcloud app browse
```

---

## 5. Docker Deployment {#docker}

**Best for**: Consistency, portability, microservices
**Works with**: Any cloud provider, local development

### Steps

#### A. Create Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### B. Create docker-compose.yml

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8501:8501"
    environment:
      - INFURA_URL=${INFURA_URL}
      - PRIVATE_KEY=${PRIVATE_KEY}
      - CONTRACT_ADDRESS=${CONTRACT_ADDRESS}
    volumes:
      - ./data:/app/data
    restart: unless-stopped
    
  # Optional: Add database
  db:
    image: postgres:14
    environment:
      POSTGRES_DB: auditor
      POSTGRES_USER: auditor
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

#### C. Build and Run

```bash
# Build image
docker build -t ethical-ai-auditor .

# Run container
docker run -p 8501:8501 \
  -e INFURA_URL="your_url" \
  -e PRIVATE_KEY="your_key" \
  ethical-ai-auditor

# Or use docker-compose
docker-compose up -d
```

#### D. Push to Registry

```bash
# Docker Hub
docker tag ethical-ai-auditor yourusername/ethical-ai-auditor:latest
docker push yourusername/ethical-ai-auditor:latest

# AWS ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com
docker tag ethical-ai-auditor YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/ethical-ai-auditor:latest
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/ethical-ai-auditor:latest
```

---

## 6. Blockchain Configuration {#blockchain}

### A. Setup Infura

1. **Sign Up**
   - Go to [infura.io](https://infura.io)
   - Create free account

2. **Create Project**
   - Dashboard → Create New Project
   - Name: "Ethical AI Auditor"
   - Select: Web3 API

3. **Get Endpoint**
   - Select "Sepolia" testnet
   - Copy HTTPS endpoint
   - Format: `https://sepolia.infura.io/v3/YOUR_PROJECT_ID`

### B. Setup Wallet

1. **Create Wallet**
   - Install MetaMask browser extension
   - Create new wallet
   - Save seed phrase securely

2. **Get Test ETH**
   - Switch to Sepolia testnet
   - Visit faucet: [sepoliafaucet.com](https://sepoliafaucet.com)
   - Request test ETH (0.5 ETH recommended)

3. **Export Private Key**
   - MetaMask → Account Details → Export Private Key
   - Enter password
   - Copy key (without 0x prefix)
   - **NEVER share or commit this key!**

### C. Deploy Smart Contract

#### Option 1: Using Remix

1. **Open Remix**
   - Go to [remix.ethereum.org](https://remix.ethereum.org)

2. **Create Contract**
   - Create new file: `EthicalAIAuditor.sol`
   - Copy contract code from `app.py` comments

3. **Compile**
   - Compiler version: 0.8.0+
   - Click "Compile"

4. **Deploy**
   - Environment: "Injected Provider - MetaMask"
   - Network: Sepolia
   - Click "Deploy"
   - Confirm transaction in MetaMask
   - Copy deployed contract address

#### Option 2: Using Hardhat

```bash
# Initialize project
npx hardhat init

# Install dependencies
npm install @openzeppelin/contracts

# Create contract
# (Copy contract code to contracts/EthicalAIAuditor.sol)

# Create deployment script
cat > scripts/deploy.js << EOF
async function main() {
  const EthicalAIAuditor = await ethers.getContractFactory("EthicalAIAuditor");
  const auditor = await EthicalAIAuditor.deploy();
  await auditor.deployed();
  console.log("Contract deployed to:", auditor.address);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
EOF

# Deploy
npx hardhat run scripts/deploy.js --network sepolia
```

### D. Configure Application

**Local Development (.env file)**
```bash
INFURA_URL=https://sepolia.infura.io/v3/YOUR_PROJECT_ID
PRIVATE_KEY=your_private_key_without_0x
CONTRACT_ADDRESS=0xYourDeployedContractAddress
```

**Streamlit Cloud (secrets.toml)**
```toml
INFURA_URL = "https://sepolia.infura.io/v3/YOUR_PROJECT_ID"
PRIVATE_KEY = "your_private_key_without_0x"
CONTRACT_ADDRESS = "0xYourDeployedContractAddress"
```

**Environment Variables (Production)**
```bash
export INFURA_URL="https://sepolia.infura.io/v3/YOUR_PROJECT_ID"
export PRIVATE_KEY="your_private_key_without_0x"
export CONTRACT_ADDRESS="0xYourDeployedContractAddress"
```

---

## Post-Deployment Checklist

- [ ] Application accessible via URL
- [ ] All features working correctly
- [ ] Sample data generation functional
- [ ] File upload working (CSV, PyTorch models)
- [ ] Audit execution completing successfully
- [ ] Visualizations rendering properly
- [ ] Reports downloadable
- [ ] Blockchain features (if enabled):
  - [ ] Connection to Infura successful
  - [ ] Transactions being logged
  - [ ] NFT minting working
  - [ ] Verification on Etherscan
- [ ] Error handling working
- [ ] Performance acceptable (<10s for audits)
- [ ] Mobile responsive (if applicable)
- [ ] SSL/HTTPS configured
- [ ] Monitoring/logging setup
- [ ] Backup strategy in place

---

## Monitoring & Maintenance

### Application Monitoring

**Streamlit Cloud**
- Built-in analytics dashboard
- Resource usage metrics
- Error tracking

**Custom Monitoring**
```python
# Add to app.py
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info("Audit started")
```

### Performance Optimization

1. **Caching**
```python
@st.cache_data
def load_large_dataset(file):
    return pd.read_csv(file)
```

2. **Resource Limits**
```python
# Limit file size
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Subsample large datasets
if len(df) > 10000:
    df = df.sample(10000)
```

3. **Async Processing**
```python
import asyncio

async def run_audit_async(df):
    # Run heavy computations asynchronously
    pass
```

---

## Security Best Practices

1. **Never commit secrets**
   - Use `.gitignore`
   - Use environment variables
   - Rotate keys regularly

2. **Input validation**
   - File size limits
   - File type checking
   - Data sanitization

3. **Rate limiting**
   - Implement request throttling
   - Use authentication for production

4. **HTTPS only**
   - Force SSL/TLS
   - HSTS headers

5. **Regular updates**
   - Keep dependencies updated
   - Monitor security advisories

---

## Troubleshooting Guide

### Common Issues

**"Module not found"**
- Verify `requirements.txt` is complete
- Check Python version compatibility
- Clear pip cache: `pip cache purge`

**"Port already in use"**
- Change port: `streamlit run app.py --server.port=8502`
- Kill process: `lsof -ti:8501 | xargs kill -9`

**"Out of memory"**
- Reduce dataset size
- Use sampling for large files
- Upgrade instance/dyno

**"Blockchain connection failed"**
- Verify Infura URL
- Check network (Sepolia)
- Ensure sufficient testnet ETH
- Validate private key format

---

## Support Resources

- **Streamlit**: [docs.streamlit.io](https://docs.streamlit.io)
- **Heroku**: [devcenter.heroku.com](https://devcenter.heroku.com)
- **AWS**: [docs.aws.amazon.com](https://docs.aws.amazon.com)
- **Docker**: [docs.docker.com](https://docs.docker.com)
- **Web3.py**: [web3py.readthedocs.io](https://web3py.readthedocs.io)

---

**Happy Deploying! 🚀**
