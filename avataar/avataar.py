import argparse
import random
import torch
import torchvision.transforms as T
from PIL import Image, ImageEnhance
import io
from rembg import remove
from torchvision import models
from diffusers import StableDiffusionPipeline
import imageio
import numpy as np
from ultralytics import YOLO
from transformers import DPTFeatureExtractor, DPTForDepthEstimation

# Load pre-trained DeepLabV3 for scene segmentation
def load_scene_segmentation_model():
    model = models.segmentation.deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1').eval()
    return model

# Load pre-trained MiDaS model for depth estimation
def load_depth_estimation_model():
    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    return model, feature_extractor

# Perform scene segmentation using DeepLabV3
def segment_scene(scene_image):
    scene_image = scene_image.convert("RGB")
    segmentation_model = load_scene_segmentation_model()

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = transform(scene_image).unsqueeze(0)
    with torch.no_grad():
        output = segmentation_model(input_tensor)['out'][0]
        segmentation_mask = output.argmax(0).cpu().numpy()

    return segmentation_mask

# Load YOLOv8 model for object detection
def load_object_detection_model():
    model = YOLO('yolov8n.pt')
    return model

# Detect objects in the scene to avoid overlaps
def detect_objects(scene_image):
    detection_model = load_object_detection_model()
    results = detection_model(scene_image)
    return results

# Get depth map using MiDaS model
def get_depth_map(scene_image):
    depth_model, feature_extractor = load_depth_estimation_model()
    inputs = feature_extractor(images=scene_image, return_tensors="pt")
    with torch.no_grad():
        outputs = depth_model(**inputs)
        depth_map = outputs.predicted_depth.squeeze().cpu().numpy()
    return depth_map

# Adjust object size based on the scene depth
def scale_object_based_on_depth(object_image, depth_map, target_depth):
    scene_height, scene_width = depth_map.shape
    avg_depth = np.mean(depth_map)
    scaling_factor = avg_depth / target_depth

    # Adjust object size based on depth and scene dimensions
    object_width, object_height = object_image.size
    new_size = (int(object_width * scaling_factor), int(object_height * scaling_factor))
    return object_image.resize(new_size)

# Adjust object brightness to match scene lighting
def adjust_lighting(object_image):
    enhancer = ImageEnhance.Brightness(object_image)
    return enhancer.enhance(1.1)

# Get a random position for placing the object in the scene
def get_random_position(scene_image, object_image):
    scene_width, scene_height = scene_image.size
    object_width, object_height = object_image.size

    if object_width > scene_width or object_height > scene_height:
        print("Object too large for scene")
        return None

    x_position = random.randint(0, scene_width - object_width)
    y_position = random.randint(0, scene_height - object_height)
    return x_position, y_position

# Place object in the scene using segmentation, depth, and object detection
def place_object_in_scene(object_image, scene_image):
    object_image = object_image.convert("RGBA")
    scene_image = scene_image.convert("RGBA")

    # Get depth map of the scene
    depth_map = get_depth_map(scene_image)
    avg_depth = np.mean(depth_map)

    # Scale the object based on depth
    scaled_object = scale_object_based_on_depth(object_image, depth_map, avg_depth)

    # Segment scene to identify regions
    segmentation_mask = segment_scene(scene_image)

    # Detect objects to avoid overlaps
    detection_results = detect_objects(scene_image)

    valid_positions = []
    for row in range(segmentation_mask.shape[0]):
        for col in range(segmentation_mask.shape[1]):
            if segmentation_mask[row, col] == 2:  # Choose region based on desired class (e.g., table/counter)
                valid_positions.append((col, row))

    # Avoid detected object positions
    for result in detection_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            valid_positions = [
                (pos_x, pos_y) for pos_x, pos_y in valid_positions
                if not (x1 < pos_x + scaled_object.width < x2 and y1 < pos_y + scaled_object.height < y2)
            ]

    if valid_positions:
        position = random.choice(valid_positions)
    else:
        position = get_random_position(scene_image, scaled_object)

    if position is None:
        print("No valid position found for the object.")
        return scene_image

    brightened_object = adjust_lighting(scaled_object)
    scene_image.paste(brightened_object, position, brightened_object)

    return scene_image

# Remove background from the object image
def remove_background(image_path):
    with open(image_path, 'rb') as i:
        input_image = i.read()
    output_image = remove(input_image)
    return Image.open(io.BytesIO(output_image)).convert("RGBA")

# Generate scene based on text prompt
def generate_scene(text_prompt):
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    generated_image = pipe(text_prompt).images[0]
    return generated_image

# Main function
def main(image_path, text_prompt, output_path, num_images):
    object_image = remove_background(image_path)

    for i in range(num_images):
        scene_image = generate_scene(text_prompt)
        final_image = place_object_in_scene(object_image, scene_image)
        final_image.save(f"{output_path}/gen1_image{i+1}.png")
        print(f"Image saved: {output_path}/gen_image{i+1}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to the object image")
    parser.add_argument("--text-prompt", type=str, required=True, help="Text prompt for generating scene")
    parser.add_argument("--output", type=str, required=True, help="Output path for the final images")
    parser.add_argument("--num-images", type=int, default=3, help="Number of scene images to generate")

    args = parser.parse_args()
    main(args.image, args.text_prompt, args.output, args.num_images)
