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
import math

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




def process_directory(dir_path, model, classes):
    json_path = Path('data/tags.json')
    
    # new data structure instead of check ing if exist
    data = {"files": {}}
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    text_extensions = {'.txt'}
    model_extensions = {'.glb'}
    
    # First process regular files
    for file in Path(dir_path).glob('*'):
        if file.is_dir():
            print(f"Processing model folder: {file.name}")
            
            # Look for .glb file in folder (using same name as folder)
            model_files = list(file.glob('*.glb'))
            if not model_files:
                print(f"No model files found in {file.name}")
                continue
                
            model_file = model_files[0]  # Use the first GLB file found
            model_rel_path = str(model_file.relative_to(dir_path))
            
            # Look for thumbnail
            thumbnail_files = list(file.glob('thumbnail.*'))
            thumbnail_rel_path = None
            if thumbnail_files:
                thumbnail_rel_path = str(thumbnail_files[0].relative_to(dir_path))
            
            # Look for tags in txt file
            tag_files = list(file.glob('*.txt'))
            model_tags = []
            if tag_files:
                model_tags = analyze_text(tag_files[0])
            
            filename = file.name.lower()
            filename_tag = {"tag": filename, "confidence": "100.00%"}
            
            creation_time = datetime.fromtimestamp(os.path.getctime(model_file))
            year = str(creation_time.year)
            month = creation_time.strftime("%B").lower()
            year_tag = {"tag": year, "confidence": "100.00%"}
            month_tag = {"tag": month, "confidence": "100.00%"}
            type_tag = {"tag": "3d", "confidence": "100.00%"}
            
            # Always use new tags, ignoring any existing tags
            final_tags = model_tags + [
                year_tag,
                month_tag,
                type_tag,
                filename_tag
            ]
            
            data["files"][model_rel_path] = {
                "type": "3d",
                "tags": final_tags,
                "last_analyzed": datetime.now().isoformat(),
                "file_size": os.path.getsize(model_file),
                "thumbnail": thumbnail_rel_path
            }
        
        elif file.suffix.lower() in image_extensions:
            rel_path = str(file.relative_to(dir_path))
            print(f"Processing image: {rel_path}")
            
            image_tensor, dimensions = prepare_image(file)
            if image_tensor is not None:
                predictions = get_image_prediction(model, classes, image_tensor)
                new_tags = [
                    {"tag": tag, "confidence": f"{conf:.2f}%"}
                    for tag, conf in predictions
                ]

                dominant_color = get_dominant_color(file)
                color_tag = {"tag": dominant_color, "confidence": "100.00%"}
                
                filename = file.stem.lower()
                filename_tag = {"tag": filename, "confidence": "100.00%"}
                
                creation_time = datetime.fromtimestamp(os.path.getctime(file))
                year = str(creation_time.year)
                month = creation_time.strftime("%B").lower()
                year_tag = {"tag": year, "confidence": "100.00%"}
                month_tag = {"tag": month, "confidence": "100.00%"}
                type_tag = {"tag": "image", "confidence": "100.00%"}
                
                # Always use new tags
                final_tags = new_tags + [
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
                
        elif file.suffix.lower() in text_extensions:
            rel_path = str(file.relative_to(dir_path))
            print(f"Processing text: {rel_path}")
            
            new_tags = analyze_text(file)
            
            filename = file.stem.lower()
            filename_tag = {"tag": filename, "confidence": "100.00%"}
            
            creation_time = datetime.fromtimestamp(os.path.getctime(file))
            year = str(creation_time.year)
            month = creation_time.strftime("%B").lower()
            year_tag = {"tag": year, "confidence": "100.00%"}
            month_tag = {"tag": month, "confidence": "100.00%"}
            type_tag = {"tag": "text", "confidence": "100.00%"}
            
            # Always use new tags
            final_tags = new_tags + [
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
        
        elif file.suffix.lower() in model_extensions:
            rel_path = str(file.relative_to(dir_path))
            print(f"Processing standalone 3D model: {rel_path}")
            
            # Look for companion text file
            txt_path = file.with_suffix('.txt')
            model_tags = []
            if txt_path.exists():
                model_tags = analyze_text(txt_path)
            
            filename = file.stem.lower()
            filename_tag = {"tag": filename, "confidence": "100.00%"}
            
            creation_time = datetime.fromtimestamp(os.path.getctime(file))
            year = str(creation_time.year)
            month = creation_time.strftime("%B").lower()
            year_tag = {"tag": year, "confidence": "100.00%"}
            month_tag = {"tag": month, "confidence": "100.00%"}
            type_tag = {"tag": "3d", "confidence": "100.00%"}
            
            # Always use new tags
            final_tags = model_tags + [
                year_tag,
                month_tag,
                type_tag,
                filename_tag
            ]
            
            data["files"][rel_path] = {
                "type": "3d",
                "tags": final_tags,
                "last_analyzed": datetime.now().isoformat(),
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
