````markdown
Customer Complaint Intelligence API  
A production-ready, end-to-end NLP classification service built with PyTorch, Sentence-BERT embeddings, FastAPI, Docker, and AWS EC2 deployment.

This project demonstrates a complete real-world ML system — from model training to cloud deployment — designed to help companies automatically analyze and categorize customer complaints (billing issues, service problems, harassment, legal risk, etc.).

It is designed as a portfolio-ready project, showcasing:
- API development and software engineering  
- ML inference pipelines  
- Docker containerization  
- Cloud deployment (AWS EC2)

---

Features

Machine Learning  
- PyTorch neural classifier  
- SBERT (`all-MiniLM-L6-v2`) text embeddings  
- Softmax probability output  
- Trained binary or multi-class classifier  
- Model exported as `model.pt`

API (FastAPI)
- POST /predict → classify a complaint  
- GET /health → health check  
- Request/response validation using Pydantic  
- Automatic OpenAPI/Swagger documentation at `/docs`

Docker
- Production image using `python:3.11-slim`  
- Clean build layers  
- Start container with:  
  ```bash
  docker run -p 8000:8000 complaint-api
````

### AWS Deployment (EC2)

* Amazon Linux 2023 instance
* Docker installed and enabled
* Image transferred via SCP
* Container exposed on port 8000
* Security Group configured for public access

---

## Project Structure

```
customer-complaint-intelligence-api/
│
├── app/
│   ├── main.py               # FastAPI entrypoint
│   ├── schemas.py            # Request/response models
│   ├── dependencies.py       # Lazy-loading ML pipeline
│   └── ...
│
├── model/
│   ├── inference.py          # PyTorch inference pipeline
│   ├── model.pt              # Trained classifier
│   └── labels.json           # Label index mapping
│
├── data/
│   └── sample_complaints.csv # Optional training sample
│
├── docker/
│   └── Dockerfile            # Production Dockerfile
│
├── requirements.txt
└── README.md
```

---

## Example Prediction

### Request

```json
{
  "text": "I am being charged incorrect fees on my credit card."
}
```

### Response

```json
{
  "label": "billing_issue",
  "probabilities": {
    "billing_issue": 0.91,
    "service_problem": 0.05,
    "other": 0.04
  }
}
```

---

## Run Locally (Uvicorn)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start the API

```bash
uvicorn app.main:app --reload
```

### Open the documentation

```
http://127.0.0.1:8000/docs
```

---

## Run with Docker

### Build the image

```bash
docker build -t complaint-api -f docker/Dockerfile .
```

### Run the container

```bash
docker run -p 8000:8000 complaint-api
```

### Test the health endpoint

```
GET http://localhost:8000/health
```

---

## Deploy on AWS EC2 (Overview)

This project can be deployed on Amazon Linux 2023 using Docker.

### 1. Launch EC2 instance

* Amazon Linux 2023
* t2.micro or t3.micro
* Root volume: 20 GB
* Security Group inbound rules:

  * SSH (22) → 0.0.0.0/0
  * Custom TCP (8000) → 0.0.0.0/0

### 2. Install Docker

```bash
sudo dnf install docker -y
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ec2-user
```

Reconnect SSH.

### 3. Transfer image to EC2

From your local machine:

```bash
scp -i "your-key.pem" complaint-api.tar ec2-user@EC2_PUBLIC_IP:/home/ec2-user/
```

### 4. Load image on EC2

```bash
docker load -i complaint-api.tar
```

### 5. Start the container

```bash
docker run -d -p 8000:8000 complaint-api
```

Access Swagger documentation at:

```
http://EC2_PUBLIC_IP:8000/docs
```

---

## Why This Project Matters

This repository demonstrates the full production lifecycle of an ML service:

* Data preprocessing and training
* PyTorch model development
* Embedding-based inference pipeline
* Serving ML models via API
* Containerization with Docker
* Cloud deployment on AWS
* Clean software architecture
* API documentation and usage examples
---

## License

MIT License.

---

## Author

Mădălina Diaconu
Machine Learning Engineer
GitHub: [https://github.com/yourusername](https://github.com/yourusername)


