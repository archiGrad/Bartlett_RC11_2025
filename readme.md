mage and Text Search Engine

A search engine that analyzes images and text files, extracting features like dominant colors, image classifications, and text topics. Results are stored in a searchable JSON format with a web interface.

## Setup

1. Create a GitHub repository and enable GitHub Pages
2. Clone the repository locally and make a first update
3. Set up Python virtual environment:

**Linux/Mac:**
```bash
python -m venv searchengine
source searchengine/bin/activate
pip install -r requirements.txt
```

**Windows:**
```bash
python -m venv searchengine
searchengine\Scripts\activate
pip install -r requirements.txt
```

### Requirements

```
torch        # Deep learning
torchvision  # Computer vision tools
Pillow      # Image processing
numpy       # Numerical computation
```

## Features

- Image Analysis:
  - Dominant color extraction using PIL
  - Image classification with ResNet50
  - Image dimensions and metadata
- Text Analysis:
  - Basic tokenization
  - Keyword extraction
  - Word frequency analysis
- Web Interface:
  - Real-time search
  - Visual results display
  - Color-coded tags
  - File previews

## File Structure

```
.
├── data/            # Directory for files to analyze
│   └── tags.json    # Generated analysis results
├── index.html       # Web interface
├── styles.css       # Styling
├── search.js        # Search functionality
├── tags.py         # Analysis engine
└── requirements.txt # Python dependencies
```

## JSON Output Format

```json
{
  "files": {
    "example_filename.png": {
      "type": "image",
      "tags": [
        {
          "tag": "architecture",
          "confidence": "75.50%"
        }
      ],
      "color": "example_color",
      "created": "YYYY-MM-DDThh:mm:ss.ssssss",
      "file_size": 123456
    }
  }
}
```

## Usage

1. Add files to analyze in the `data` directory
2. Run the analysis:
```bash
python tags.py
```
3. Open `index.html` in a browser or deploy to GitHub Pages
4. Search files using tags in the search interface

## Web Interface

The interface includes:
- Search bar for tag-based queries
- Grid display of results
- File previews (images/text)
- Color indicators for images
- Confidence scores for tags

## Implementation Details

The project consists of three main components:

### 1. Analysis Engine (tags.py)
- Image processing with PIL and ResNet50
- Text analysis with basic NLP
- JSON data generation

### 2. Frontend (HTML/CSS)
- Responsive grid layout
- Clean, modern design
- Preview capabilities

### 3. Search (JavaScript)
- Real-time search functionality
- Multi-tag support
- Dynamic result display
