# Image and Text Search Engine

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
3. Open up a python http.server in the working directory to see the webpage. `python -m http.server`
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






## python workflow

# setup ml model
def setup_model():
    """
    Sets up a pre-trained ResNet50 model.
    
    The model comes pre-trained on ImageNet (a dataset of ~1.2 million images and 1000 classes).
    We use the default weights and put the model in evaluation mode since we're not training.
    """
    # Load pre-trained ResNet50 with the latest weights
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    
    # Set model to evaluation mode (disables dropout layers and batch normalization)
    model.eval()
    
    # Get the class labels that the model was trained on
    class_labels = weights.meta["categories"]
    
    return model, class_labels




# 2. Preparing Images
def prepare_image(image_path):
    """
    Prepares an image for the ResNet model by:
    1. Resizing to 256x256
    2. Center cropping to 224x224 (standard input size for ResNet)
    3. Converting to tensor
    4. Normalizing with ImageNet statistics
    
    Args:
        image_path: Path to the image file
    Returns:
        Tensor ready for model input and original image dimensions
    """
    try:
        # Open image file
        img = Image.open(image_path)
        
        # Define transformation pipeline
        transform = transforms.Compose([
            # Resize the shortest side to 256 while maintaining aspect ratio
            transforms.Resize(256),
            
            # Take 224x224 center crop
            transforms.CenterCrop(224),
            
            # Convert PIL image to tensor (changes values to 0-1 range)
            transforms.ToTensor(),
            
            # Normalize using ImageNet statistics
            # These specific values are the mean and std of ImageNet
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # RGB means
                std=[0.229, 0.224, 0.225]    # RGB standard deviations
            )
        ])
        
        # Apply transformations
        img_tensor = transform(img)
        
        # Add batch dimension (ResNet expects batch_size x channels x height x width)
        batch_tensor = torch.unsqueeze(img_tensor, 0)
        
        return batch_tensor, img.size
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None



# 3. Making Predictions
def get_image_prediction(model, classes, image_tensor):
    """
    Gets top 5 predictions for an image.
    
    Args:
        model: The ResNet model
        classes: List of class labels
        image_tensor: Prepared image tensor
    Returns:
        List of (class_name, confidence) tuples
    """
    # Disable gradient calculation for inference
    with torch.no_grad():
        # Get raw model outputs
        outputs = model(image_tensor)
    
    # Sort predictions by confidence (descending)
    # Returns tuple of (values, indices)
    _, indices = torch.sort(outputs, descending=True)
    
    # Convert raw outputs to probabilities with softmax
    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
    
    # Get top 5 predictions with their probabilities
    top5_predictions = [
        (classes[idx], probabilities[idx].item())
        for idx in indices[0][:5]
    ]
    
    return top5_predictions















