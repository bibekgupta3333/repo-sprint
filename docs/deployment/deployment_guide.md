# Deployment Guide
# LLM Agentic Sprint Intelligence Platform

**Version**: 1.0.0  
**Last Updated**: February 14, 2026  
**Target Platform**: macOS (M4 Pro), Linux, Windows (with Docker)

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start (5 Minutes)](#quick-start-5-minutes)
3. [Detailed Setup](#detailed-setup)
4. [Configuration](#configuration)
5. [Running the Application](#running-the-application)
6. [Troubleshooting](#troubleshooting)
7. [Production Deployment](#production-deployment)

---

## Prerequisites

### System Requirements

| Component | Requirement | Recommended |
|-----------|-------------|-------------|
| **OS** | macOS 12+, Ubuntu 20.04+, Windows 10+ | macOS 14 (Sonoma) |
| **CPU** | 4+ cores | 8+ cores (M4 Pro) |
| **RAM** | 16GB minimum | 24GB+ |
| **Storage** | 50GB free | 100GB+ SSD |
| **Docker** | 4.15+ | Latest |
| **Docker Compose** | 2.0+ | Latest |

### Software Dependencies

#### 1. Install Docker Desktop

**macOS**:
```bash
# Download from https://www.docker.com/products/docker-desktop/
# Or use Homebrew
brew install --cask docker
```

**Linux (Ubuntu)**:
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
    -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

**Windows**:
- Download Docker Desktop from [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
- Enable WSL 2 backend

#### 2. Install Git

```bash
# macOS
brew install git

# Linux
sudo apt-get install git

# Windows - Download from https://git-scm.com/
```

#### 3. GitHub Account & Personal Access Token

**Create GitHub Personal Access Token**:
1. Go to [https://github.com/settings/tokens](https://github.com/settings/tokens)
2. Click "Generate new token (classic)"
3. Select scopes:
   - ✅ `repo` (Full control of private repositories)
   - ✅ `read:org` (Read org memberships)
   - ✅ `workflow` (Update GitHub Action workflows)
4. Copy the token (save it securely!)

---

## Quick Start (5 Minutes)

### Step 1: Clone Repository

```bash
git clone https://github.com/bibekgupta3333/repo-sprint.git
cd repo-sprint
```

### Step 2: Create Environment Configuration

```bash
# Copy template
cp .env.example .env

# Edit .env file
nano .env
```

**Minimal `.env` configuration**:
```bash
# GitHub API
GITHUB_TOKEN=your_github_personal_access_token_here
GITHUB_ORG=your-org-name  # e.g., "microsoft"

# Database
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres123
POSTGRES_DB=sprint_intelligence

# LLM
OLLAMA_MODEL=llama3:8b-q4

# API
API_SECRET_KEY=your-secret-key-change-this-in-production
FRONTEND_URL=http://localhost:8501
```

### Step 3: Start Services

```bash
# Pull and start all containers
docker-compose up -d

# Wait for services to be healthy (~2 minutes)
docker-compose ps
```

### Step 4: Initialize Database

```bash
# Run migrations
docker-compose exec fastapi-backend alembic upgrade head

# Seed initial data (optional)
docker-compose exec fastapi-backend python scripts/seed_data.py
```

### Step 5: Pull LLM Model

```bash
# Pull Llama-3-8B quantized model (~5GB download)
docker-compose exec ollama-server ollama pull llama3:8b-q4
```

### Step 6: Access Application

Open your browser:
- **Frontend (Streamlit)**: [http://localhost:8501](http://localhost:8501)
- **Backend API (FastAPI)**: [http://localhost:8000](http://localhost:8000)
- **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Detailed Setup

### Project Structure

```
repo-sprint/
├── apps/
│   ├── backend/              # FastAPI application
│   │   ├── api/              # API routes
│   │   ├── services/         # Business logic
│   │   ├── models/           # Database models
│   │   ├── core/             # Core configs
│   │   └── requirements.txt
│   └── frontend/             # Streamlit application
│       ├── app.py
│       ├── pages/
│       ├── components/
│       └── requirements.txt
├── docker-compose.yml        # Service orchestration
├── .env                      # Environment variables
├── data/                     # Data storage
│   ├── raw/                  # GitHub Archive data
│   └── processed/            # Embeddings, ChromaDB
├── doc/                      # Documentation
└── scripts/                  # Utility scripts
```

### Docker Compose Services

```yaml
# docker-compose.yml (overview)
services:
  streamlit-app:        # Frontend (port 8501)
  fastapi-backend:      # Backend API (port 8000)
  postgres-db:          # PostgreSQL (port 5432)
  chromadb:             # Vector store (port 8001)
  redis-cache:          # Cache & queue (port 6379)
  ollama-server:        # LLM inference (port 11434)
```

### Service Dependencies

```mermaid
graph TB
    START([docker-compose up])
    
    START --> PG[postgres-db starts]
    START --> REDIS[redis-cache starts]
    START --> CHROMA[chromadb starts]
    START --> OLLAMA[ollama-server starts]
    
    PG --> BACKEND[fastapi-backend starts]
    REDIS --> BACKEND
    CHROMA --> BACKEND
    OLLAMA --> BACKEND
    
    BACKEND --> FRONTEND[streamlit-app starts]
    
    FRONTEND --> READY([Application Ready])
    
    classDef storage fill:#10b981,stroke:#059669,color:#fff
    classDef app fill:#6366f1,stroke:#4f46e5,color:#fff
    
    class PG,REDIS,CHROMA,OLLAMA storage
    class BACKEND,FRONTEND app
```

---

## Configuration

### Environment Variables Reference

#### `.env` File (Complete Template)

```bash
# =================================
# GitHub Configuration
# =================================
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GITHUB_ORG=your-organization-name
GITHUB_REPOS=repo1,repo2,repo3  # Comma-separated (or leave empty for all)
GITHUB_WEBHOOK_SECRET=your_webhook_secret_for_security

# =================================
# PostgreSQL Database
# =================================
POSTGRES_USER=postgres
POSTGRES_PASSWORD=secure_password_change_me
POSTGRES_DB=sprint_intelligence
POSTGRES_HOST=postgres-db
POSTGRES_PORT=5432
DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}

# =================================
# ChromaDB (Vector Store)
# =================================
CHROMA_HOST=chromadb
CHROMA_PORT=8001
CHROMA_PERSIST_PATH=/chroma/data

# =================================
# Redis (Cache & Queue)
# =================================
REDIS_HOST=redis-cache
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0

# =================================
# Ollama LLM
# =================================
OLLAMA_HOST=ollama-server
OLLAMA_PORT=11434
OLLAMA_MODEL=llama3:8b-q4
OLLAMA_BASE_URL=http://${OLLAMA_HOST}:${OLLAMA_PORT}

# =================================
# Embedding Model
# =================================
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# =================================
# FastAPI Backend
# =================================
API_HOST=0.0.0.0
API_PORT=8000
API_SECRET_KEY=generate_a_secure_random_key_here
API_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440  # 24 hours
CORS_ORIGINS=http://localhost:8501,http://127.0.0.1:8501

# =================================
# Streamlit Frontend
# =================================
FRONTEND_URL=http://localhost:8501
BACKEND_URL=http://localhost:8000

# =================================
# Logging
# =================================
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT=json  # json or text

# =================================
# Feature Flags
# =================================
ENABLE_SYNTHETIC_DATA=true
ENABLE_REAL_TIME_ANALYSIS=true
ENABLE_CROSS_REPO_ANALYSIS=true

# =================================
# Performance Tuning
# =================================
MAX_WORKERS=4
BATCH_SIZE=100
CACHE_TTL=3600  # 1 hour in seconds
```

### Generating Secret Keys

```bash
# Generate API_SECRET_KEY
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate GITHUB_WEBHOOK_SECRET
openssl rand -hex 32
```

---

## Running the Application

### Development Mode

```bash
# Start all services in foreground (see logs)
docker-compose up

# Start in background (detached mode)
docker-compose up -d

# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f fastapi-backend

# Restart a service
docker-compose restart fastapi-backend

# Stop all services
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v
```

### Health Checks

```bash
# Check service status
docker-compose ps

# Expected output:
# NAME                COMMAND             SERVICE          STATUS            PORTS
# streamlit-app      "streamlit run..."   streamlit-app    Up 2 minutes     0.0.0.0:8501->8501/tcp
# fastapi-backend    "uvicorn main:app"   fastapi-backend  Up 2 minutes     0.0.0.0:8000->8000/tcp
# postgres-db        "docker-entrypoint"  postgres-db      Up 2 minutes     0.0.0.0:5432->5432/tcp
# chromadb           "uvicorn chroma..."  chromadb         Up 2 minutes     0.0.0.0:8001->8000/tcp
# redis-cache        "redis-server"       redis-cache      Up 2 minutes     0.0.0.0:6379->6379/tcp
# ollama-server      "ollama serve"       ollama-server    Up 2 minutes     0.0.0.0:11434->11434/tcp

# Test backend API
curl http://localhost:8000/health
# Expected: {"status": "healthy"}

# Test Ollama
curl http://localhost:11434/api/tags
# Expected: List of available models
```

### Database Management

```bash
# Access PostgreSQL shell
docker-compose exec postgres-db psql -U postgres -d sprint_intelligence

# Run migrations
docker-compose exec fastapi-backend alembic upgrade head

# Create new migration
docker-compose exec fastapi-backend alembic revision --autogenerate -m "Add new table"

# Rollback migration
docker-compose exec fastapi-backend alembic downgrade -1

# Backup database
docker-compose exec postgres-db pg_dump -U postgres sprint_intelligence > backup_$(date +%Y%m%d).sql

# Restore database
cat backup_20260214.sql | docker-compose exec -T postgres-db psql -U postgres -d sprint_intelligence
```

### Data Collection

```bash
# Collect GitHub data for organization
docker-compose exec fastapi-backend python scripts/collect_github_data.py \
    --org your-org-name \
    --repos repo1,repo2 \
    --days 30

# Download GitHub Archive data
docker-compose exec fastapi-backend python scripts/download_github_archive.py \
    --sample \
    --max-files 24

# Generate embeddings
docker-compose exec fastapi-backend python scripts/prepare_embeddings.py \
    --max-files 1 \
    --max-docs-per-file 100
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Port Already in Use

**Error**: `Bind for 0.0.0.0:8501 failed: port is already allocated`

**Solution**:
```bash
# Find process using port
lsof -i :8501  # macOS/Linux
netstat -ano | findstr :8501  # Windows

# Kill process
kill -9 <PID>

# Or change port in docker-compose.yml
```

#### Issue 2: Ollama Model Not Found

**Error**: `model 'llama3:8b-q4' not found`

**Solution**:
```bash
# List available models
docker-compose exec ollama-server ollama list

# Pull model manually
docker-compose exec ollama-server ollama pull llama3:8b-q4

# Use alternative model
OLLAMA_MODEL=llama3:8b docker-compose up -d
```

#### Issue 3: Database Connection Error

**Error**: `FATAL: password authentication failed for user "postgres"`

**Solution**:
```bash
# Ensure .env matches docker-compose.yml
# Rebuild containers
docker-compose down -v
docker-compose up -d

# Check PostgreSQL logs
docker-compose logs postgres-db
```

#### Issue 4: Out of Memory

**Error**: `Container killed (OOMKilled)`

**Solution**:
```bash
# Increase Docker memory limit in Docker Desktop Settings
# Recommended: 16GB+ for Docker

# Or reduce services:
# Stop Ollama and use OpenAI API instead
docker-compose stop ollama-server
```

#### Issue 5: ChromaDB Errors

**Error**: `chromadb.errors.InvalidDimensionException`

**Solution**:
```bash
# Clear ChromaDB data and re-embed
docker-compose down
rm -rf data/processed/chromadb/*
docker-compose up -d

# Re-run embedding script
docker-compose exec fastapi-backend python scripts/prepare_embeddings.py
```

### Logs & Debugging

```bash
# Enable debug logging
# Edit .env
LOG_LEVEL=DEBUG

# Restart services
docker-compose restart

# Follow all logs
docker-compose logs -f --tail=100

# Export logs
docker-compose logs > debug_logs_$(date +%Y%m%d).txt
```

### Performance Monitoring

```bash
# Check resource usage
docker stats

# Example output:
# CONTAINER       CPU %   MEM USAGE / LIMIT    MEM %    NET I/O
# ollama-server   15%     5.2GiB / 24GiB       21.67%   1.2kB / 890B
# fastapi-backend 2%      500MiB / 24GiB       2.08%    10kB / 5kB
# postgres-db     1%      150MiB / 24GiB       0.63%    2kB / 1kB
```

---

## Production Deployment

### Security Hardening

#### 1. Environment Variables

```bash
# NEVER commit .env to Git
echo ".env" >> .gitignore

# Use strong passwords
POSTGRES_PASSWORD=$(openssl rand -base64 32)
API_SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
```

#### 2. HTTPS/TLS

```yaml
# Add Nginx reverse proxy to docker-compose.yml
nginx:
  image: nginx:alpine
  ports:
    - "443:443"
    - "80:80"
  volumes:
    - ./nginx.conf:/etc/nginx/nginx.conf
    - ./ssl:/etc/nginx/ssl
  depends_on:
    - fastapi-backend
    - streamlit-app
```

#### 3. Rate Limiting

```python
# apps/backend/api/middleware/rate_limit.py
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/api/v1/analysis")
@limiter.limit("10/minute")
async def analyze_milestone():
    ...
```

### Scaling

#### Horizontal Scaling (Multiple Replicas)

```bash
# Scale backend to 3 replicas
docker-compose up -d --scale fastapi-backend=3

# Add load balancer (e.g., Nginx, HAProxy)
```

#### Vertical Scaling (Resource Limits)

```yaml
# docker-compose.yml
services:
  fastapi-backend:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### Monitoring & Alerts

```bash
# Add Prometheus + Grafana
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
```

### Backup Automation

```bash
# Add to crontab
0 2 * * * /path/to/repo-sprint/scripts/backup_postgres.sh
0 3 * * 0 /path/to/repo-sprint/scripts/backup_chromadb.sh
```

---

## Next Steps

After successful deployment:

1. ✅ **Configure GitHub Webhooks**:
   - Go to GitHub repo → Settings → Webhooks
   - Payload URL: `https://your-domain.com/api/v1/webhooks/github`
   - Secret: (from `.env` `GITHUB_WEBHOOK_SECRET`)
   - Events: Issues, Pull Requests, Push

2. ✅ **Add Target Repositories**:
   - Log in to Streamlit app
   - Go to Settings → Add Repository

3. ✅ **Collect Historical Data**:
   - Run `scripts/collect_github_data.py` for past 6 months

4. ✅ **Set Up Monitoring**:
   - Configure alerts for high latency, errors
   - Monitor GitHub API rate limits

5. ✅ **Invite Team Members**:
   - Share Streamlit URL with project managers

---

## Support

**Issues**: [https://github.com/bibekgupta3333/repo-sprint/issues](https://github.com/bibekgupta3333/repo-sprint/issues)  
**Documentation**: [doc/](../README.md)  
**Contact**: bibekgupta3333@github.com

---

**Document Version**: 1.0.0  
**Last Updated**: February 14, 2026  
**Tested On**: macOS 14 (M4 Pro), Ubuntu 22.04, Docker 24.0.7
