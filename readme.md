# Image and Text Search Engine

A search engine that analyzes images and text files, extracting features like dominant colors, image classifications, and text topics. Results are stored in a searchable JSON format with a web interface.

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


## Setup

1. Create a GitHub repository and enable GitHub Pages with a simple index.html file
2. Set up Python virtual environment:


### Requirements

```
torch        # Deep learning
torchvision  # Computer vision tools
Pillow      # Image processing
numpy       # Numerical computation
```


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

# Setting Up ResNet50 for Auto-Tagging Images

## 1. Preparing Images
The image preparation process involves several key steps to make images compatible with ResNet:

### Image Preparation Steps
1. Resizing to 256x256
2. Center cropping to 224x224 (standard input size for ResNet)
3. Converting to tensor
4. Normalizing with ImageNet statistics

### Tensor Conversion Details
- Images start as pixels with RGB values (0-255)
- A tensor is a multi-dimensional array optimized for neural networks
- Changes pixel values from 0-255 to 0-1 range
- Rearranges data from (height, width, channels) to (channels, height, width)
- Makes the data compatible with PyTorch operations

### Normalization Explanation
- ResNet was trained on ImageNet, which has specific statistical properties
- Normalization adjusts our image to match these properties
- Ensures:
  - The input data has similar properties to what the model expects
  - The model's internal calculations work optimally
  - More consistent and reliable predictions

## 2. Implementation Code

### Image Preparation Function
```python
def prepare_image(image_path):
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
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # RGB means
                std=[0.229, 0.224, 0.225]    # RGB standard deviations
            )
        ])
        
        # Apply transformations
        img_tensor = transform(img)
        
        # Add batch dimension
        batch_tensor = torch.unsqueeze(img_tensor, 0)
        
        return batch_tensor, img.size
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None
```

### Return Values Explained

#### batch_tensor
- Format: 4-dimensional tensor (batch_size, channels, height, width)
- Dimensions:
  - batch_size = 1 (single image processing)
  - channels = 3 (RGB channels)
  - height = 224 (from crop)
  - width = 224 (from crop)
- Final shape: (1, 3, 224, 224)

**Tensor Shape Transformation:**
- Before unsqueeze: (3, 224, 224)
- After unsqueeze: (1, 3, 224, 224)

#### img.size
- Contains original image dimensions (width, height)
- Example: (800, 600) for an 800x600 pixel image
- Uses:
  - Reference purposes
  - Scaling predictions back
  - Maintaining aspect ratio information
  - Metadata storage

## 3. Model Setup
```python
def setup_model():
    # Load pre-trained ResNet50 with latest weights
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    
    # Set evaluation mode
    model.eval()
    
    # Get class labels
    class_labels = weights.meta["categories"]
    
    return model, class_labels
```

## 4. Making Predictions
```python
def get_image_prediction(model, classes, image_tensor):
    # Disable gradient calculation for inference
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Sort predictions by confidence
    _, indices = torch.sort(outputs, descending=True)
    
    # Convert to probabilities
    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
    
    # Get top 5 predictions
    top5_predictions = [
        (classes[idx], probabilities[idx].item())
        for idx in indices[0][:5]
    ]
    
    return top5_predictions
```




# Color Analysis Functions
# These functions extract and identify the dominant color from images by comparing RGB values
# to a predefined color palette using Euclidean distance in RGB space.

```python
def get_closest_color_name(rgb):
    colors = {
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'black': (0, 0, 0),
        'white': (255, 255, 255),
        'yellow': (255, 255, 0),
        'purple': (128, 0, 128),
        'orange': (255, 165, 0),
        'brown': (165, 42, 42),
        'pink': (255, 192, 203),
        'gray': (128, 128, 128)
    }
    
    r, g, b = rgb
    distances = {}
    for color_name, color_rgb in colors.items():
        distance = math.sqrt(
            (r - color_rgb[0]) ** 2 +
            (g - color_rgb[1]) ** 2 +
            (b - color_rgb[2]) ** 2
        )
        distances[color_name] = distance
    
    return min(distances, key=distances.get)

def get_dominant_color(image_path):
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img = img.resize((150, 150))
            paletted = img.quantize(colors=1)
            palette = paletted.getpalette()
            dominant_rgb = (palette[0], palette[1], palette[2])
            color_name = get_closest_color_name(dominant_rgb)
            return color_name
            
    except Exception as e:
        print(f"Error getting color from {image_path}: {e}")
        return "unknown"
```

# Text Analysis Function
# A simple text tokenizer that cleans text, removes common words,
# and returns the top 5 most frequent meaningful words with their percentages.
### Processing Steps:
1. Text Cleanup:
   - Converts to lowercase
   - Removes punctuation
   - Splits into words
   - Removes stop words
   - Keeps alphanumeric words
   - Removes words ≤ 2 characters

2. Analysis:
   - Counts word frequency
   - Calculates word percentages
   - Returns top 5 frequent words
This function scans a directory for images and text files, processes each file to generate tags (using image recognition for images and text analysis for text files), and saves all the metadata (tags, file info, creation dates, colors for images) into a JSON file at 'data/tags.json'. For images, it generates tags based on what the AI model recognizes, dominant colors, and creation dates, while for text files it extracts key words and metadata - all of this creates a searchable database of file information.


```python
def analyze_text(text_path):
    try:
        with open(text_path, 'r', encoding='utf-8') as file:
            text = file.read().lower()
        
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        words = [word for word in words if 
                word not in stop_words and 
                word.isalnum() and 
                len(word) > 2]
        
        word_freq = Counter(words)
        total_words = sum(word_freq.values()) or 1
        top_words = word_freq.most_common(5)
        
        return [
            {"tag": word, "confidence": f"{(count/total_words*100):.2f}%"}
            for word, count in top_words
        ]
    except Exception as e:
        print(f"Error processing text file {text_path}: {e}")
        return []
```




# Directory Processing Function
# Scans a directory for images and text files, processes each file to generate tags,
# and saves all metadata (tags, file info, creation dates, colors) into a JSON file.

### Function Overview:
1. Loads/creates JSON storage
2. Recursively scans directory
3. For each file:
   - Extracts basic metadata (name, date)
   - Processes images: gets ML predictions, color, dimensions
   - Processes texts: analyzes content
   - Adds tags and metadata to JSON
4. Saves all data to tags.json

```python
def process_directory(dir_path, model, classes):
    json_path = Path('data/tags.json')
    
    data = {"files": {}}
    
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                content = f.read().strip()
                if content:
                    data = json.loads(content)
                    if "images" in data and "files" not in data:
                        data["files"] = data["images"]
                        del data["images"]
                    elif "files" not in data:
                        data["files"] = {}
        except json.JSONDecodeError:
            print("Warning: Invalid JSON file, starting fresh")
            data = {"files": {}}
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    text_extensions = {'.txt'}
    
    for file in Path(dir_path).rglob('*'):
        rel_path = str(file.relative_to(dir_path))
        file_ext = file.suffix.lower()
        # Get filename without extension as tag
        filename = file.stem.lower()
        filename_tag = {"tag": filename, "confidence": "100.00%"}
        
        creation_time = datetime.fromtimestamp(os.path.getctime(file))
        year = str(creation_time.year)
        month = creation_time.strftime("%B").lower()
        year_tag = {"tag": year, "confidence": "100.00%"}
        month_tag = {"tag": month, "confidence": "100.00%"}
        
        if file_ext in image_extensions:
            print(f"Processing image: {rel_path}")
            image_tensor, dimensions = prepare_image(file)
            if image_tensor is not None:
                existing_entry = data["files"].get(rel_path, {})
                existing_tags = existing_entry.get("tags", [])
                predictions = get_image_prediction(model, classes, image_tensor)
                new_tags = [
                    {"tag": tag, "confidence": f"{conf:.2f}%"}
                    for tag, conf in predictions
                ]
                dominant_color = get_dominant_color(file)
                color_tag = {"tag": dominant_color, "confidence": "100.00%"}
                type_tag = {"tag": "image", "confidence": "100.00%"}
                
                final_tags = existing_tags if existing_tags else new_tags + [
                    color_tag,
                    year_tag,
                    month_tag,
                    type_tag,
                    filename_tag
                ]
                
                data["files"][rel_path] = {
                    "type": "image",
                    "tags": final_tags,
                    "color": dominant_color,
                    "dimensions": f"{dimensions[0]}x{dimensions[1]}",
                    "last_analyzed": datetime.now().isoformat(),
                    "file_size": os.path.getsize(file)
                }
                
        elif file_ext in text_extensions:
            print(f"Processing text: {rel_path}")
            existing_entry = data["files"].get(rel_path, {})
            existing_tags = existing_entry.get("tags", [])
            new_tags = analyze_text(file)
            type_tag = {"tag": "text", "confidence": "100.00%"}
            
            final_tags = existing_tags if existing_tags else new_tags + [
                year_tag,
                month_tag,
                type_tag,
                filename_tag
            ]
            
            data["files"][rel_path] = {
                "type": "text",
                "tags": final_tags,
                "last_analyzed": datetime.now().isoformat(),
                "file_size": os.path.getsize(file)
            }
    json_path.parent.mkdir(exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
```

# Main Program Entry Point
# Sets up the ResNet model and processes all files in the 'data' directory
### What this does:
1. Initializes the ML model when script runs
2. Processes all files in 'data' directory
3. Entry point check prevents automatic execution if imported as module



```python
def main():
    print("Setting up model...")
    model, classes = setup_model()
    
    print("Processing files...")
    process_directory('data', model, classes)
    
    print("Done! Results saved to data/tags.json")

# Program entry point - only runs if script is executed directly
if __name__ == "__main__":
    main()
```












