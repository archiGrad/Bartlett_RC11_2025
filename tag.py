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
