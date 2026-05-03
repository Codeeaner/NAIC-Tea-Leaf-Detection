---
title: Tea Leaf Detection Website
emoji: 🍃
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
---

# Tea Leaf Detection Website

A comprehensive web application for AI-powered tea leaf health detection using YOLOv8. This system allows users to upload images of tea leaves and get instant analysis of leaf health status with detailed reports.

## Features

- **Single & Batch Image Processing**: Upload individual images or process multiple images in batches
- **AI-Powered Detection**: Uses YOLOv11 model to detect and classify healthy vs unhealthy tea leaves
- **🆕 AI Analytics with llava-phi3:3.8b**: Advanced image analysis and waste prevention recommendations
- **🆕 Waste Prevention System**: Intelligent recommendations to minimize tea leaf waste
- **Real-time Progress Tracking**: Monitor processing progress with live updates
- **Detection History**: View and manage all previous detection sessions
- **Comprehensive Reports**: Generate PDF and CSV reports with detailed analysis
- **Interactive Web Interface**: Modern, responsive design with drag-and-drop functionality
- **RESTful API**: Full API access with automatic documentation
- **🆕 Quality Metrics Tracking**: Monitor quality trends and processing efficiency

## Technology Stack

- **Backend**: FastAPI, SQLAlchemy, Pydantic
- **Frontend**: HTML5, Bootstrap 5, JavaScript
- **Machine Learning**: YOLOv11 (Ultralytics), OpenCV
- **AI Analytics**: llava-phi3:3.8b via Ollama

## 🆕 AI Analytics & Waste Prevention

The Tea Leaf Detection system now includes advanced AI-powered analytics using **llava-phi3:3.8b** to provide intelligent waste prevention recommendations and quality insights.

### Analytics Features

- **Image Analysis**: Advanced AI vision analysis of detection results
- **Waste Prevention**: Actionable recommendations to minimize leaf waste
- **Quality Assessment**: Automated quality grading and severity analysis
- **Processing Guidance**: Specific instructions for handling defective leaves
- **Economic Impact**: Cost-saving estimates and value recovery methods
- **Batch Analytics**: Aggregate analysis across multiple detection sessions

### What Analytics Provides

1. **Overall Assessment**
   - Quality grade (premium/standard/below_standard/reject)
   - Severity level (low/medium/high)
   - Health percentage analysis

2. **Processing Recommendations**
   - Immediate actions to take
   - Sorting strategies
   - Quality preservation methods
   - Processing sequence optimization

3. **Waste Prevention Strategies**
   - Salvageable portion identification
   - Alternative uses for defective leaves
   - Composting guidelines
   - Prevention measures for future harvests

4. **Economic Impact Analysis**
   - Estimated loss percentage
   - Cost-saving opportunities
   - Value recovery methods
   - ROI optimization suggestions

### Setup Analytics

1. **Install Ollama** (see [`setup_ollama.md`](setup_ollama.md) for detailed instructions)
   ```bash
   # Download and install Ollama from https://ollama.com
   ollama serve
   ```

2. **Install llava-phi3:3.8b Model**
   ```bash
   ollama pull llava-phi3:3.8b
   ```

3. **Test Analytics**
   ```bash
   python test_analytics.py
   ```

### Analytics API Endpoints

- `POST /api/analytics/analyze/{result_id}` - Analyze single detection result
- `POST /api/analytics/analyze/batch/{session_id}` - Analyze batch results
- `GET /api/analytics/results/{analysis_id}` - Get analysis results
- `GET /api/analytics/recommendations/{analysis_id}` - Get waste prevention recommendations
- `GET /api/analytics/history` - View analytics history

- **Database**: SQLite (development), PostgreSQL (production ready)
- **Reports**: ReportLab (PDF), Pandas (CSV), Matplotlib (charts)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd tea-leaf-website
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**
   - Copy `.env.example` to `.env` (if needed)
   - Update the `MODEL_PATH` in `.env` to point to your trained model:
     ```
     MODEL_PATH=./runs/detect/train/weights/best.pt
     ```
   - If you publish the model to Hugging Face Spaces, set the Space URL so the homepage button opens it:
     ```
       HUGGINGFACE_SPACE_URL=https://huggingface.co/spaces/your-username/your-space
     ```
    - To load the YOLO weights from Hugging Face, set the model repo details too:
       ```
       HUGGINGFACE_MODEL_REPO_ID=your-username/your-space
       HUGGINGFACE_MODEL_FILE=best.pt
       HUGGINGFACE_MODEL_REPO_TYPE=space
       HUGGINGFACE_MODEL_REVISION=main
       ```
    - If your weights are in a Hugging Face model repo instead of a Space, set `HUGGINGFACE_MODEL_REPO_TYPE=model`.

## Publish the YOLO Model to Hugging Face Spaces

1. **Create a new Space**
   - In Hugging Face, create a Space for your project.
   - Use a Docker Space if you want to run this FastAPI app directly, or a Gradio/Streamlit Space if you only want a simple model demo.

2. **Upload the trained weights**
   - Add your YOLO weights file, such as `best.pt`, to the Space repository or store it as a large file with Git LFS.
   - Keep the same class order and class names that your app expects.

3. **Add inference code**
   - Load the weights with Ultralytics YOLO in the Space.
   - Expose an input image upload and return the prediction results.

4. **Set environment variables in the Space**
   - Add any required config values, such as confidence threshold or model path.
   - Make sure the app can find the uploaded weights file.

5. **Copy the Space URL into `.env`**
   - Put the published Space URL into `HUGGINGFACE_SPACE_URL`.
   - The homepage will show an "Open Hugging Face Space" button that links there.

5. **Initialize Database**
   ```bash
   python -c "from app.database import create_tables; create_tables()"
   ```

6. **Create Required Directories**
   ```bash
   mkdir uploads results reports static
   ```

## Running the Application

### Development Mode

```bash
# Run with auto-reload
python app/main.py

# Or use uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

The application will be available at:
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc

## Usage

### Web Interface

1. **Upload Images**
   - Navigate to the home page
   - Enter a session name
   - Drag and drop images or click to browse
   - Adjust confidence threshold if needed
   - Click "Start Detection"

2. **Monitor Progress**
   - Real-time progress bar shows processing status
   - Automatic results display when complete

3. **View Results**
   - Detailed statistics (healthy/unhealthy counts, percentages)
   - Individual image results with bounding boxes
   - Download PDF or CSV reports

4. **History Management**
   - View all previous detection sessions
   - Filter by status, date, or search by name
   - Delete sessions to manage storage

### API Usage

#### Upload Single Image

```bash
curl -X POST "http://localhost:8000/api/upload/single" \
  -H "Content-Type: multipart/form-data" \
  -F "session_name=My Test Session" \
  -F "confidence_threshold=0.25" \
  -F "file=@/path/to/image.jpg"
```

#### Upload Multiple Images

```bash
curl -X POST "http://localhost:8000/api/upload/batch" \
  -H "Content-Type: multipart/form-data" \
  -F "session_name=Batch Test" \
  -F "confidence_threshold=0.25" \
  -F "files=@/path/to/image1.jpg" \
  -F "files=@/path/to/image2.jpg"
```

#### Get Results

```bash
curl -X GET "http://localhost:8000/api/results/{session_id}"
```

#### Download Reports

```bash
# PDF Report
curl -X GET "http://localhost:8000/api/reports/download/pdf/{session_id}" -o report.pdf

# CSV Report
curl -X GET "http://localhost:8000/api/reports/download/csv/{session_id}" -o data.csv
```

## API Endpoints

### Upload Endpoints
- `POST /api/upload/single` - Upload single image
- `POST /api/upload/batch` - Upload multiple images
- `GET /api/upload/progress/{session_id}` - Get processing progress
- `DELETE /api/upload/session/{session_id}` - Delete session

### Detection Endpoints
- `GET /api/results/{session_id}` - Get detailed results
- `GET /api/sessions` - List all sessions with pagination
- `GET /api/history` - Get recent sessions
- `GET /api/stats` - Get overall statistics
- `GET /api/sessions/{session_id}/summary` - Get session summary

### Report Endpoints
- `POST /api/reports/generate/{session_id}` - Generate reports
- `GET /api/reports/download/pdf/{session_id}` - Download PDF
- `GET /api/reports/download/csv/{session_id}` - Download CSV
- `GET /api/reports/status/{session_id}` - Get report status

## Model Information

The application uses a YOLOv8 model trained to detect tea leaf health with two classes:
- **Class 0**: Unhealthy leaves
- **Class 1**: Healthy leaves

### Model Requirements

- Model file: `best.pt` (YOLOv8 format)
- Input: RGB images
- Classes: 2 (healthy, unhealthy)
- Confidence threshold: Configurable (default 0.25)

## Configuration

### Environment Variables

```env
DATABASE_URL=sqlite:///./tea_leaf_detection.db
MODEL_PATH=./runs/detect/train/weights/best.pt
CONFIDENCE_THRESHOLD=0.25
MAX_FILE_SIZE=10485760  # 10MB
ALLOWED_EXTENSIONS=.jpg,.jpeg,.png,.webp
MAX_BATCH_SIZE=50
UPLOAD_DIR=uploads
RESULTS_DIR=results
REPORTS_DIR=reports
```

### File Limits

- Maximum file size: 10MB per image
- Supported formats: JPG, JPEG, PNG, WEBP
- Maximum batch size: 50 images per upload

## Project Structure

```
tea-leaf-website/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── database.py             # Database configuration
│   ├── schemas.py              # Pydantic models
│   ├── models/                 # Database models
│   │   ├── __init__.py
│   │   └── detection.py
│   ├── services/               # Business logic
│   │   ├── __init__.py
│   │   ├── detection_service.py
│   │   ├── batch_service.py
│   │   └── report_service.py
│   └── api/                    # API endpoints
│       ├── __init__.py
│       ├── upload.py
│       ├── detection.py
│       └── reports.py
├── templates/                  # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── history.html
│   └── error.html
├── static/                     # Static files
├── uploads/                    # Uploaded images
├── results/                    # Detection results
├── reports/                    # Generated reports
├── requirements.txt
├── .env
└── README.md
```

## Troubleshooting

### Common Issues

1. **Model not found error**
   - Ensure the model file exists at the specified path
   - Check MODEL_PATH in .env file
   - Verify model is in YOLOv8 format

2. **Database errors**
   - Run database initialization: `python -c "from app.database import create_tables; create_tables()"`
   - Check database permissions

3. **Upload failures**
   - Verify file size is under 10MB
   - Check file format is supported
   - Ensure upload directory exists and is writable

4. **Processing stuck**
   - Check server logs for errors
   - Verify model can be loaded
   - Restart the application

### Performance Optimization

1. **For large batches**
   - Process in smaller chunks
   - Consider using background task queues (Celery)
   - Monitor memory usage

2. **For production deployment**
   - Use PostgreSQL instead of SQLite
   - Set up reverse proxy (nginx)
   - Configure proper logging
   - Use multiple workers

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest
```

### Code Style

```bash
# Install formatting tools
pip install black isort flake8

# Format code
black app/
isort app/

# Check style
flake8 app/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please:
1. Check this README for common issues
2. Review the API documentation at `/docs`
3. Check application logs for error details
4. Create an issue in the repository

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments

- YOLOv8 by Ultralytics for object detection
- FastAPI for the web framework
- Bootstrap for the frontend design
