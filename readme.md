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
├── imagenet_classes.txt         
└── requirements.txt # Python dependencies
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

# Auto-Tagging System Documentation

## Table of Contents
1. [ResNet50 Setup](#resnet50-setup)
2. [Color Analysis](#color-analysis)
3. [Text Analysis](#text-analysis)
4. [Directory Processing](#directory-processing)
5. [Main Program](#main-program)
6. [Frontend Implementation](#frontend-implementation)



## ResNet50 Setup

### 1. Preparing Images
The image preparation process involves several key steps to make images compatible with ResNet:

#### Image Preparation Steps
1. Resizing to 256x256
2. Center cropping to 224x224 (standard input size for ResNet)
3. Converting to tensor
4. Normalizing with ImageNet statistics

#### Tensor Conversion Details
- Images start as pixels with RGB values (0-255)
- A tensor is a multi-dimensional array optimized for neural networks
- Changes pixel values from 0-255 to 0-1 range
- Rearranges data from (height, width, channels) to (channels, height, width)
- Makes the data compatible with PyTorch operations

#### Normalization Explanation
- ResNet was trained on ImageNet, which has specific statistical properties
- Normalization adjusts our image to match these properties
- Ensures:
  - The input data has similar properties to what the model expects
  - The model's internal calculations work optimally
  - More consistent and reliable predictions

#### Return Values Explained
- **batch_tensor**:
  - Format: 4-dimensional tensor (batch_size, channels, height, width)
  - Dimensions: (1, 3, 224, 224)
  - Shape transformation:
    - Before unsqueeze: (3, 224, 224)
    - After unsqueeze: (1, 3, 224, 224)

- **img.size**:
  - Contains original dimensions (width, height)
  - Example: (800, 600)
  - Used for reference and metadata



### Implementation Code

```python
def prepare_image(image_path):
    try:
        # Open image file
        img = Image.open(image_path)
        
        # Define transformation pipeline
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Apply transformations
        img_tensor = transform(img)
        batch_tensor = torch.unsqueeze(img_tensor, 0)
        
        return batch_tensor, img.size
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None
```
### Model Setup Code

 
# Model Setup and Prediction

## 1. Model Setup
The model setup process prepares ResNet50 for image classification:

### Setup Steps
1. Loading pre-trained weights
2. Setting up the model
3. Preparing for inference
4. Loading class labels

### Model Details
- Uses ResNet50 architecture
- Pre-trained on ImageNet dataset
- Contains 50 layers of deep learning
- Capable of recognizing 1000 different classes as seen in imagenet_classes.txt

### Components Explained
- **weights**:
  - Uses DEFAULT weights from ImageNet
  - Contains millions of pre-trained parameters
  - Represents learned features from vast image dataset

- **class_labels**:
  - List of 1000 categories
  - Matches model's output dimensions
  - Used to convert numerical predictions to human-readable labels

## 2. Making Predictions
The prediction process converts image tensors into classification results:

### Prediction Steps
1. Running inference without gradients
2. Sorting predictions by confidence
3. Converting to probabilities
4. Selecting top 5 results

### Components Explained
- **torch.no_grad()**:
  - Disables gradient calculation

- **model(image_tensor)**:
  - Processes the image through all 50 layers
  - Shape: (1, 1000) for single image

- **Sorting and Probabilities**:
  - Sort outputs by confidence (descending)
  - Convert logits to percentages using softmax
  - Scale to 0-100 range for readability

### Return Format
```python
[
    ("category1", 95.2),  # (class_name, confidence_percentage)
    ("category2", 82.1),
    ("category3", 76.5),
    ("category4", 65.8),
    ("category5", 45.2)
]
```

### Implementation Code
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




## Color Analysis
Functions to extract and identify dominant colors from images.

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

## Text Analysis
Text tokenizer that processes content and extracts key terms.

### Processing Steps
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
            {"tag": word, "confidence":""}
            for word, count in top_words
        ]
    except Exception as e:
        print(f"Error processing text file {text_path}: {e}")
        return []
```

## Directory Processing
Central function that processes files and generates the tag database.

### Overview
1. Loads/creates JSON storage
2. Recursively scans directory
3. For each file:
   - Extracts basic metadata (name, date)
   - Processes images: gets ML predictions, color, dimensions
   - Processes texts: analyzes content
   - Adds tags and metadata to JSON
4. Saves all data to tags.json with the following file structure.

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

## Main Program
Entry point of the application.

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




## Frontend Implementation

### 1. User Interface Structure
The frontend consists of three main components:
1. HTML structure for the search interface
2. CSS styling for visual presentation
3. JavaScript for search functionality and tag display

## frontend

### html
``` html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>File Search</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="search-container">
        <input 
            type="text" 
            id="searchInput" 
            placeholder="Search files by tags..."
        >
        <div id="results" class="results-container"></div>
    </div>
    <script src="search.js"></script>
</body>
</html>
```


### style.css

``` css
body {
    margin: 0;
    padding: 20px;
    font-family: monospace ;
	background: black;
	color: white;
}

.search-container {
    margin:0  100px;
}

#searchInput {
    width: 100%;
    max-width: 600px;
    padding: 15px 20px;
    font-size: 16px;
    outline: none;
    margin: 20px auto 40px;
    display: block;
	color: white;
	background: black;
	border-radius: 30px;
	border: 1px solid grey;
}


.results-container {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 20px;
    padding: 20px 0;
}


.result-item:hover {
    background: rgb(30,30,30);
}

.file-preview {
    position: relative;
    height: 300px;
    overflow: hidden;
}

.file-preview img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.text-content {
    height: 200px;
    padding: 15px;
    overflow-y: auto;
    font-family: monospace;
    Font-size: 14px;
    line-height: 1.4;
}

.result-info {
    padding: 15px;
}

.file-path {
    font-size: 14px;
    margin-bottom: 8px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.tags {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 8px;
}

.tag {
    padding: 2px 8px;
    font-size: 12px;
}

.color-indicator {
    display: flex;
    align-items: center;
    margin-top: 8px;
    font-size: 12px;
}

.color-dot {
    width: 12px;
    height: 12px;
    margin-right: 6px;
	border-radius:12px;
}

.file-type {
    padding: 2px 8px;
	color:red;
}




.result-item {
    cursor: pointer;
}

.result-item:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}








```


### search.js

``` javascript
let fileData = null;
let fileContents = {};


const colorMap = {
    'red': '#FF0000',
    'green': '#008000',
    'blue': '#0000FF',
    'black': '#000000',
    'white': '#FFFFFF',
    'yellow': '#FFFF00',
    'purple': '#800080',
    'orange': '#FFA500',
    'brown': '#A52A2A',
    'pink': '#FFC0CB',
    'gray': '#808080',
    'unknown': '#CCCCCC'
};

async function loadTextContents() {
    for (const [path, file] of Object.entries(fileData.files)) {
        if (file.type === 'text') {
            try {
                const response = await fetch('data/' + path);
                fileContents[path] = await response.text();
            } catch (error) {
                console.error(`Error loading text file ${path}:`, error);
                fileContents[path] = 'Error loading file content';
            }
        }
    }
}


async function loadData() {
    try {
        const response = await fetch('data/tags.json');
        fileData = await response.json();
        console.log('Data loaded:', fileData);
        // Pre-load text contents
        await loadTextContents();
    } catch (error) {
        console.error('Error loading JSON:', error);
    }
}


function searchFiles(query) {
    if (!fileData || !query) return [];
    
    query = query.toLowerCase();
    const searchTerms = query.split(/\s+/); // Split on whitespace
    const results = [];
    
    for (const [path, file] of Object.entries(fileData.files)) {
        // Check if all search terms match
        const hasAllTerms = searchTerms.every(term => {
            return file.tags.some(tagObj => {
                const tagText = tagObj.tag.toLowerCase();
                // Check if the tag contains the term or if multiple tags together match the term
                return tagText.includes(term) || 
                       file.tags.map(t => t.tag.toLowerCase()).join(' ').includes(term);
            });
        });
        
        if (hasAllTerms) {
            results.push({
                path,
                ...file
            });
        }
    }
    
    return results;
}


function displayResults(results) {
    const resultsContainer = document.getElementById('results');
    
    if (results.length === 0) {
        resultsContainer.style.display = 'none';
        return;
    }
    
    resultsContainer.style.display = 'grid';
    resultsContainer.innerHTML = results.map(result => `
        <div class="result-item" onclick="openFile('data/${result.path}')">
            <div class="file-preview">
                ${getFilePreview(result)}
            </div>
            <div class="result-info">
                <div class="file-path">${result.path}</div>
                ${result.type === 'image' ? `
                    <div class="color-indicator">
                        <span class="color-dot" style="background-color: ${colorMap[result.color] || result.color}"></span>
                        <span>${result.color}</span>
                    </div>
                ` : ''}
                <div class="tags">
                    ${result.tags.map(tag => 
                        `<span class="tag">${tag.tag} (${tag.confidence})</span>`
                    ).join('')}
                </div>
            </div>
        </div>
    `).join('');
}

function openFile(path) {
    window.open(path, '_blank');
}

function getFilePreview(result) {
    if (result.type === 'image') {
        return `<img src="data/${result.path}" alt="${result.path}">`;
    } else if (result.type === 'text') {
        const content = fileContents[result.path] || 'Loading...';
        return `<div class="text-content">${escapeHtml(content)}</div>`;
    }
    return '';
}

function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

document.getElementById('searchInput').addEventListener('input', (e) => {
    const query = e.target.value.trim();
    const results = searchFiles(query);
    displayResults(results);
});

loadData();
```




# Update 2: Comprehensive 3D Model Support Implementation
- go ever the code changes and explain
- implement it our selves
- make our own glb files
- upload it to the website


## Global Changes
- Added support for 3D model file type (.glb)
- Implemented 3D model preview and viewing functionality
- Enhanced file search and display to support 3D models

## GLB files
GLB (GL Transmission Format Binary) is a file format for 3D models that stores geometry, materials, textures, and animations in a single compact, binary file. It's based on the glTF (GL Transmission Format) standard, which is designed for efficient transmission and loading of 3D content on the web and in applications. GLB files are widely supported across different 3D rendering platforms, making them ideal for web-based 3D experiences, games, and interactive visualizations. They offer a good balance between file size and model complexity, supporting features like PBR (Physically Based Rendering) materials and complex scene hierarchies.
we will use mainly blender for any file conversion.

sketchfab.com will also give you good search results.

## three.js
Three.js is a popular, lightweight, and cross-browser JavaScript library used for creating and displaying 3D computer graphics in a web browser. Developed by Ricardo Cabello (mrdoob), it provides an abstraction layer over WebGL, making it much easier to create 3D graphics without needing to write complex WebGL code directly. Three.js allows developers to create sophisticated 3D visualizations, animations, games, and interactive experiences using simple JavaScript, with built-in support for loading various 3D model formats, adding lights and cameras, creating materials and geometries, and implementing complex rendering techniques. Its extensive documentation, large community, and consistent updates make it a go-to library for web-based 3D graphics.


# code changes supporting the additions


## python tag generation updates
we need to make sure the .glb files are stored in a folder (arbitarary name) containing a "thumbnail".png/jpg/gif with a text file ("name".txt) containing the hashtags

```py
# Added 3D model file extensions
model_extensions = {'.glb'}

def process_directory(dir_path, model, classes):
    for file in Path(dir_path).glob('*'):
        # Process 3D model folders
        if file.is_dir():
            # Look for .glb files
            model_files = list(file.glob('*.glb'))
            if not model_files:
                continue
            
            model_file = model_files[0]
            model_rel_path = str(model_file.relative_to(dir_path))
            
            # Find thumbnails
            thumbnail_files = list(file.glob('thumbnail.*'))
            thumbnail_rel_path = None
            if thumbnail_files:
                thumbnail_rel_path = str(thumbnail_files[0].relative_to(dir_path))
            
            # Extract tags from companion text files
            tag_files = list(file.glob('*.txt'))
            model_tags = []
            if tag_files:
                model_tags = analyze_text(tag_files[0])
            
            # Add to file data
            data["files"][model_rel_path] = {
                "type": "3d",
                "tags": model_tags,
                "last_analyzed": datetime.now().isoformat(),
                "file_size": os.path.getsize(model_file),
                "thumbnail": thumbnail_rel_path
            }
        
        # Process standalone 3D model files
        elif file.suffix.lower() in model_extensions:
            rel_path = str(file.relative_to(dir_path))
            
            # Look for companion text file for tags
            txt_path = file.with_suffix('.txt')
            model_tags = []
            if txt_path.exists():
                model_tags = analyze_text(txt_path)
            
            # Add to file data
            data["files"][rel_path] = {
                "type": "3d",
                "tags": model_tags,
                "last_analyzed": datetime.now().isoformat(),
                "file_size": os.path.getsize(file)
            }

```




## index.html Modifications
```html
<!-- Added Three.js library imports -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r126/three.min.js" integrity="sha512-n8IpKWzDnBOcBhRlHirMZOUvEq2bLRMuJGjuVqbzUJwtTsgwOgK5aS0c1JA647XWYfqvXve8k3PtZdzpipFjgg==" crossorigin="anonymous"></script>
<script src="https://unpkg.com/three@0.126.0/examples/js/loaders/GLTFLoader.js"></script>
<script src="https://unpkg.com/three@0.126.0/examples/js/controls/OrbitControls.js"></script>


```
###  examples can be found 
https://threejs.org/examples/
https://threejs.org/docs/#examples/en/controls/OrbitControls

## search.js file modifications

```js
// Enhanced displayResults to support 3D models
function displayResults(results) {
    resultsContainer.innerHTML = results.map(result => `
        <div class="result-item">
            <!-- Added file type display -->
            <div class="file-type">${result.type}</div>
            
            <!-- Added 3D model view button -->
            ${result.type === '3d' ? `
                <button class="view-3d-btn" data-path="${result.path}">View 3D Model</button>
            ` : ''}
        </div>
    `);

    // Add event listeners for 3D model buttons
    document.querySelectorAll('.view-3d-btn').forEach(button => {
        button.addEventListener('click', function() {
            const modelPath = this.getAttribute('data-path');
            openModelViewer(modelPath);
        });
    });
}

// Enhanced getFilePreview to support 3D models
function getFilePreview(result) {
    if (result.type === '3d') {
        // Support for 3D model thumbnails
        if (result.thumbnail) {
            return `<div class="model-placeholder">
                        <img src="data/${result.thumbnail}" alt="${result.path}" class="model-thumbnail">
                    </div>`;
        }
    }
}

// New functions for 3D model viewing
function openModelViewer(modelPath) {
    let modal = document.getElementById('model-viewer-modal');
    if (!modal) {
        // Create modal dynamically if it doesn't exist
        modal = document.createElement('div');
        modal.id = 'model-viewer-modal';
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-content">
                <span class="close-modal">&times;</span>
                <div id="model-container"></div>
            </div>
        `;
        document.body.appendChild(modal);
        
        modal.querySelector('.close-modal').addEventListener('click', closeModelViewer);
    }
    
    modal.style.display = 'block';
    initThreeJsViewer('data/' + modelPath);
}

function closeModelViewer() {
    const modal = document.getElementById('model-viewer-modal');
    if (modal) {
        modal.style.display = 'none';
    }
    
    // Clean up Three.js resources
    if (currentModel) {
        currentModel.dispose();
        currentModel = null;
    }
}

function initThreeJsViewer(modelUrl) {
    // Comprehensive Three.js model viewer
    const container = document.getElementById('model-container');
    container.innerHTML = '';
    
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    
    const renderer = new THREE.WebGLRenderer({ 
        antialias: true,
        alpha: true
    });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);
    
    // Lighting and controls setup
    const ambientLight = new THREE.AmbientLight(0xcccccc, 0.4);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);
    
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    
    // Model loading logic
    const loader = new THREE.GLTFLoader();
    loader.load(modelUrl, (gltf) => {
        scene.add(gltf.scene);
        currentModel = gltf.scene;
        
    });
}
```


## css modifications

```css
/* 3D Model styles */
.model-placeholder {
    /* New styles for placeholder when 3D model is not loaded */
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: #f0f0f0;
    height: 100%;
    color: #666;
    font-size: 14px;
}


/* View 3D button */
.view-3d-btn {
    display: block;
    margin-top: 10px;
    padding: 6px 12px;
    background: #4a90e2;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 14px;
    transition: background 0.2s;
}

.view-3d-btn:hover {
    background: #357ABD;  -> hexadecimal color
}

/* Modal styles for 3D model viewer */
.modal-content {
    background-color: transparent;
    margin: 5% auto;
    width: 90%;
    max-width: 900px;
    height: 80%;
    border-radius: 10px;
    box-shadow: none;
}

#model-container {
    width: 100%;
    height: 100%;
    overflow: hidden;
    border-radius: 10px;
    background-color: transparent;
}

.modal {
    background-color: rgba(0,0,0,0.3);
}

.close-modal {
    position: absolute;
    top: 10px;
    right: 15px;
    color: #fff;
    font-size: 28px;
    font-weight: bold;
    z-index: 1001;
    cursor: pointer;
    text-shadow: 0 0 5px rgba(0,0,0,0.5);
}

.model-thumbnail {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.view-3d-overlay {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background: rgba(0, 0, 0, 0.6);
    color: white;
    padding: 8px;
    text-align: center;
    font-size: 14px;
    opacity: 0;
    transition: opacity 0.2s;
}

.model-placeholder:hover .view-3d-overlay {
    opacity: 1;
}

```
