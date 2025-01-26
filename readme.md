



making a searchengine.

make a venv with the correct requirements

1) make venv

requirements.txt
`
torch #deep learning
torchvision #computer vision tools
Pillow #image processing
numpy #numerical computation
`
linux: 
`
	python -m venv searchengine 
	source searchengine/bin/activate
	pip install -r requirements.txt
`
windows:

`
	python -m venv searchengine 
	searchengine\Scripts\activate
	pip install -r requirements.txt
`

now we have a python working enviroment

lets process the text and images and store their information in a json file.


we would like functions that can extract :


`
get_dominant_color() #we can use PIL for this.
get_image_prediction() #get the class and confidence  percentage 
analyze_text() #make a simple text tokenizer that can analyze text

`

the results will be stored in a json file of the following format


`
	{
	  "files": {
	    "example_filename.png": {
	      "type": "image",
	      "tags": [
		{
		  "tag": "example_tag",
		  "confidence": "75.50%"
		},
		{
		  "tag": "required_tag",
		  "confidence": "100.00%"
		}
	      ],
	      "color": "example_color",
	      "dimensions": "widthxheight",
	      "created": "YYYY-MM-DDThh:mm:ss.ssssss",
	      "file_size": 123456
	    }
	  }
	}

`



---------------------------


full tags.py file

`
import torch
from torchvision import models, transforms
from PIL import Image
from torchvision.models import ResNet50_Weights
import os
import json
from datetime import datetime
from pathlib import Path
from collections import Counter
import string
import re
import colorsys
import math

def get_closest_color_name(rgb):
    # Dictionary of basic colors and their RGB values
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
    
    # Calculate Euclidean distance to each known color
    r, g, b = rgb
    distances = {}
    for color_name, color_rgb in colors.items():
        distance = math.sqrt(
            (r - color_rgb[0]) ** 2 +
            (g - color_rgb[1]) ** 2 +
            (b - color_rgb[2]) ** 2
        )
        distances[color_name] = distance
    
    # Return the name of the closest color
    return min(distances, key=distances.get)

def get_dominant_color(image_path):
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize image to speed up processing
            img = img.resize((150, 150))
            
            # Quantize to get dominant color
            paletted = img.quantize(colors=1)
            palette = paletted.getpalette()
            
            # Get RGB of dominant color
            dominant_rgb = (palette[0], palette[1], palette[2])
            
            # Convert to color name
            color_name = get_closest_color_name(dominant_rgb)
            return color_name
            
    except Exception as e:
        print(f"Error getting color from {image_path}: {e}")
        return "unknown"

def setup_model():
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.eval()
    return model, weights.meta["categories"]

def prepare_image(image_path):
    try:
        img = Image.open(image_path)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        img_t = transform(img)
        batch_t = torch.unsqueeze(img_t, 0)
        return batch_t, img.size
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None

def get_image_prediction(model, classes, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
    
    _, indices = torch.sort(outputs, descending=True)
    percentages = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
    return [(classes[idx], percentages[idx].item()) for idx in indices[0][:5]]

def analyze_text(text_path):
    try:
        with open(text_path, 'r', encoding='utf-8') as file:
            text = file.read().lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Simple word tokenization
        words = text.split()
        
        # Basic stopwords
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Filter words
        words = [word for word in words if 
                word not in stop_words and 
                word.isalnum() and 
                len(word) > 2]
        
        # Get word frequency
        word_freq = Counter(words)
        
        # Get top 5 words with their frequencies
        total_words = sum(word_freq.values()) or 1
        top_words = word_freq.most_common(5)
        
        # Convert frequencies to percentages
        return [
            {"tag": word, "confidence": f"{(count/total_words*100):.2f}%"}
            for word, count in top_words
        ]
    except Exception as e:
        print(f"Error processing text file {text_path}: {e}")
        return []








def process_directory(dir_path, model, classes):
    json_path = Path('data/tags.json')
    
    # Initialize empty data structure
    data = {"files": {}}
    
    # Load existing data handling as before...
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
        
        # Get file creation time
        creation_time = datetime.fromtimestamp(os.path.getctime(file))
        year = str(creation_time.year)
        year_tag = {"tag": year, "confidence": "100.00%"}
        
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
                    type_tag
                ]
                
                data["files"][rel_path] = {
                    "type": "image",
                    "tags": final_tags,
                    "color": dominant_color,
                    "dimensions": f"{dimensions[0]}x{dimensions[1]}",
                    "created": creation_time.isoformat(),  # Using creation time instead of current time
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
                type_tag
            ]
            
            data["files"][rel_path] = {
                "type": "text",
                "tags": final_tags,
                "created": creation_time.isoformat(),  # Using creation time instead of current time
                "file_size": os.path.getsize(file)
            }

    json_path.parent.mkdir(exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)




def main():
    print("Setting up model...")
    model, classes = setup_model()
    
    print("Processing files...")
    process_directory('data', model, classes)
    
    print("Done! Results saved to data/tags.json")

if __name__ == "__main__":
    main()


`


this will result in data/tags.json 



now lets build the frontend.

a basic frontend consists of html, css and js.
html delivers the content, css delivers the syling and js delivers interaction.

html

`
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


`

the js will make it interactive


`
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
        <div class="result-item">
            <div class="file-preview">
                ${getFilePreview(result)}
            </div>
            <div class="result-info">
                <div class="file-type">${result.type}</div>
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

`

and the css will style the content


`



body {
    margin: 0;
    padding: 20px;
    min-height: 100vh;
    background-color: #f5f5f5;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
}

.search-container {
    max-width: 1200px;
    margin: 0 auto;
}

#searchInput {
    width: 100%;
    max-width: 600px;
    padding: 15px 20px;
    font-size: 16px;
    border: 2px solid #ddd;
    border-radius: 24px;
    outline: none;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
    margin: 20px auto 40px;
    display: block;
}

#searchInput:focus {
    border-color: #4a90e2;
    box-shadow: 0 0 10px rgba(74, 144, 226, 0.1);
}

.results-container {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 20px;
    padding: 20px 0;
}

.result-item {
    background: white;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s, box-shadow 0.2s;
}

.result-item:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.file-preview {
    position: relative;
    height: 200px;
    overflow: hidden;
    background: #f8f9fa;
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
    font-size: 14px;
    line-height: 1.4;
    background: #f8f9fa;
}

.result-info {
    padding: 15px;
}

.file-path {
    font-size: 14px;
    color: #666;
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
    background: #e8f0fe;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 12px;
    color: #1a73e8;
}

.color-indicator {
    display: flex;
    align-items: center;
    margin-top: 8px;
    font-size: 12px;
    color: #666;
}

.color-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 6px;
    border: 1px solid rgba(0,0,0,0.1);
}

.file-type {
    font-size: 12px;
    padding: 2px 8px;
    border-radius: 12px;
    background: #e0e0e0;
    color: #333;
}



`






