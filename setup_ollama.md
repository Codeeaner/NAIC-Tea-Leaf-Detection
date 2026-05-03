# Setting Up Ollama with Llama 3.2 Vision for Tea Leaf Analytics

This guide will help you set up Ollama with the Llama 3.2 Vision 11B model to enable AI-powered analytics for tea leaf detection results.

## Prerequisites

- Windows 10/11 or Linux/macOS
- At least 16GB RAM (32GB recommended for 11B model)
- GPU with at least 8GB VRAM (optional but recommended)

## Installation Steps

### 1. Install Ollama

#### Windows:
1. Download Ollama from: https://ollama.com/download/windows
2. Run the installer and follow the setup wizard
3. Ollama will be available as a Windows service

#### Linux/macOS:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Start Ollama Service

#### Windows:
- Ollama should start automatically as a Windows service
- If not, run: `ollama serve` in Command Prompt/PowerShell

#### Linux/macOS:
```bash
# Start Ollama service
ollama serve
```

### 3. Pull Llama 3.2 Vision Model

Open a new terminal/command prompt and run:

```bash
# Pull the Llama 3.2 Vision 11B model
ollama pull llama3.2-vision:11b

# Alternative: If you have limited resources, use the smaller 1B model
# ollama pull llama3.2-vision:1b
```

**Note:** The 11B model download is approximately 7-8GB and may take some time depending on your internet connection.

### 4. Verify Installation

Test that the model is working:

```bash
# List installed models
ollama list

# Test the vision model with a simple query
ollama run llama3.2-vision:11b "Describe this image" --image path/to/test/image.jpg
```

### 5. Configure Tea Leaf Analytics

1. Ensure Ollama is running on `http://localhost:11434` (default)
2. The Tea Leaf Detection system will automatically connect to Ollama
3. Analytics will be available through the `/api/analytics/` endpoints

## Configuration Options

### Environment Variables

Create or update your `.env` file:

```env
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2-vision:11b

# Analytics Settings
ENABLE_ANALYTICS=true
ANALYTICS_AUTO_TRIGGER=false
```

### Model Selection

If you need to use a different model size due to resource constraints:

- `llama3.2-vision:1b` - Smaller, faster, less accurate (requires ~2GB RAM)
- `llama3.2-vision:11b` - Larger, slower, more accurate (requires ~8GB RAM)

Update the model name in `app/services/analytics_service.py`:

```python
self.model_name = "llama3.2-vision:1b"  # or "llama3.2-vision:11b"
```

## Usage

### Manual Analytics

1. Run detection on your tea leaf images
2. Call the analytics endpoint: `POST /api/analytics/analyze/{result_id}`
3. Get results: `GET /api/analytics/results/{analysis_id}`

### Batch Analytics

1. Complete a batch detection session
2. Call: `POST /api/analytics/analyze/batch/{session_id}`
3. Get results: `GET /api/analytics/batch/{batch_analysis_id}`

### Enable Auto-Analytics

To automatically run analytics after each detection:

```python
from app.services.detection_service import TeaLeafDetectionService

detection_service = TeaLeafDetectionService()
detection_service.enable_analytics_mode()
```

## Troubleshooting

### Ollama Not Starting
- **Windows:** Check Windows Services for "Ollama" service
- **Linux/macOS:** Run `ollama serve` manually
- Check if port 11434 is available: `netstat -an | grep 11434`

### Model Not Found
- Verify model is pulled: `ollama list`
- Re-pull the model: `ollama pull llama3.2-vision:11b`

### Out of Memory
- Try the smaller 1B model: `ollama pull llama3.2-vision:1b`
- Close other applications to free up RAM
- Consider using a machine with more RAM

### API Connection Issues
- Verify Ollama is running: `curl http://localhost:11434/api/tags`
- Check firewall settings
- Ensure no other service is using port 11434

### Performance Issues
- **GPU Acceleration:** Ollama automatically uses GPU if available
- **CPU Only:** Performance will be slower but functional
- **Model Size:** Use smaller model for better performance

## API Examples

### Analyze Single Detection Result

```bash
curl -X POST "http://localhost:8000/api/analytics/analyze/1" \
  -H "Content-Type: application/json"
```

### Get Analysis Results

```bash
curl "http://localhost:8000/api/analytics/results/analysis_abc123_1234567890"
```

### Get Recommendations

```bash
curl "http://localhost:8000/api/analytics/recommendations/analysis_abc123_1234567890"
```

## Expected Output

The analytics will provide:

1. **Overall Assessment:** Quality grade and severity level
2. **Processing Recommendations:** Immediate actions and sorting strategies
3. **Waste Prevention:** Alternative uses and composting guidelines
4. **Economic Impact:** Cost savings and value recovery methods
5. **Priority Actions:** Top recommendations with timelines

## Support

If you encounter issues:

1. Check the application logs in the `logs/` directory
2. Verify Ollama logs: `ollama logs` (if available)
3. Test Ollama directly with a simple image analysis
4. Check system resources (RAM, disk space)

For more information, visit:
- [Ollama Documentation](https://ollama.com/docs)
- [Llama 3.2 Vision Model Info](https://ollama.com/library/llama3.2-vision)