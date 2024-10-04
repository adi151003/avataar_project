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

# Load pre-trained DeepLabV3 for scene segmentation
def load_scene_segmentation_model():
    model = models.segmentation.deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1').eval()
    return model

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

# Load pre-trained YOLOv8 model for object detection
def load_object_detection_model():
    model = YOLO('yolov8n.pt')
    return model

# Detect objects in the scene to avoid overlaps
def detect_objects(scene_image):
    detection_model = load_object_detection_model()
    results = detection_model(scene_image)
    return results

# Adjust object size based on the scene dimensions
def scale_object(object_image, scene_size):
    scene_width, scene_height = scene_size
    object_width, object_height = object_image.size

    # Scaling the object to be 15-25% of the scene dimensions
    max_scale = min(0.25 * scene_width / object_width, 0.25 * scene_height / object_height)
    if max_scale < 1:  # Only resize if necessary
        new_size = (int(object_width * max_scale), int(object_height * max_scale))
        return object_image.resize(new_size)
    
    return object_image

# Adjust object brightness to match the scene lighting
def adjust_lighting(object_image):
    enhancer = ImageEnhance.Brightness(object_image)
    return enhancer.enhance(1.1)  # Adjust brightness as needed

# Get a random position for placing the object in the scene
def get_random_position(scene_image, object_image):
    scene_width, scene_height = scene_image.size
    object_width, object_height = object_image.size

    # Ensure object dimensions are valid
    if object_width > scene_width or object_height > scene_height:
        print(f"Object too large for scene: {object_width}x{object_height} vs {scene_width}x{scene_height}")
        return None  # Indicate that a valid position could not be found

    # Random positions ensuring the object is placed within bounds
    x_position = random.randint(0, scene_width - object_width)
    y_position = random.randint(0, scene_height - object_height)
    
    return x_position, y_position

# Refined object placement logic using segmentation and object detection
def place_object_in_scene(object_image, scene_image):
    object_image = object_image.convert("RGBA")
    scene_image = scene_image.convert("RGBA")

    # Scale the object before any checks
    scaled_object = scale_object(object_image, scene_image.size)

    # Segment the scene to identify different regions
    segmentation_mask = segment_scene(scene_image)

    # Detect other objects in the scene to avoid overlaps
    detection_results = detect_objects(scene_image)

    valid_positions = []

    # Check for valid positions based on segmentation mask
    for row in range(segmentation_mask.shape[0]):
        for col in range(segmentation_mask.shape[1]):
            # If the segmentation mask suggests this is a valid placement region (e.g., table or counter)
            if segmentation_mask[row, col] == 2:  # Adjust this according to the classes in your model
                valid_positions.append((col, row))

    # Avoid placing the object near detected objects
    for result in detection_results:
        for box in result.boxes:  # Access the boxes from the detection results
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get the bounding box coordinates
            for position in valid_positions:
                pos_x, pos_y = position
                # Check if the new object's position overlaps with the detected box
                if (x1 < pos_x + scaled_object.width < x2) or (x1 < pos_x < x2):
                    valid_positions.remove(position)

    # Randomly select a valid position if any are found
    if valid_positions:
        selected_position = random.choice(valid_positions)
        position = (selected_position[0], selected_position[1])
    else:
        position = get_random_position(scene_image, scaled_object)  # Default random placement
    
    if position is None:
        print("No valid position found for the object.")
        return scene_image  # Return the scene image without modification if no position is valid

    brightened_object = adjust_lighting(scaled_object)

    # Paste the object in the selected position
    scene_image.paste(brightened_object, position, brightened_object)
    
    return scene_image

# Remove background from the object image
def remove_background(image_path):
    with open(image_path, 'rb') as i:
        input_image = i.read()
    output_image = remove(input_image)
    image_without_bg = Image.open(io.BytesIO(output_image)).convert("RGBA")
    return image_without_bg

# Generate the scene image based on the text prompt
def generate_scene(text_prompt):
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")  # Use GPU for faster generation
    generated_image = pipe(text_prompt).images[0]
    return generated_image

# Create a sequence of images for the video
def generate_video_frames(object_image, text_prompt, num_frames, output_path):
    frames = []
    
    for i in range(num_frames):
        # Generate a scene image based on the text prompt
        scene_image = generate_scene(text_prompt)
        
        # Place the object in a logical position in the scene
        final_image = place_object_in_scene(object_image, scene_image)
        
        # Save each frame as part of the video generation
        frames.append(final_image)
    
    return frames

# Save video with slower frame rate (3-5 seconds per image)
def save_video(frames, output_path, video_size=(640, 640), fps=0.2):
    video_path = f"{output_path}/generated_video.mp4"

    # Resize all frames to the same size (e.g., 640x640)
    resized_frames = [frame.resize(video_size) for frame in frames]

    # Convert frames to RGB if necessary
    rgb_frames = [frame.convert("RGB") for frame in resized_frames]

    # Save frames as video using ffmpeg writer with low FPS (for 3-5 seconds per frame)
    with imageio.get_writer(video_path, fps=fps, codec='libx264') as writer:
        for frame in rgb_frames:
            writer.append_data(np.array(frame))

    print(f"Video saved: {video_path}")

# Main function to run the pipeline
def main(image_path, text_prompt, output_path, num_images, num_frames):
    # Remove background from the object image
    object_image = remove_background(image_path)

    # Collect frames for video generation
    frames = []

    for i in range(num_images):
        # Generate a new scene image based on the text prompt
        scene_image = generate_scene(text_prompt)
        
        # Place the object logically in the scene
        final_image = place_object_in_scene(object_image, scene_image)
        
        # Save the final image with a unique name
        final_image.save(f"{output_path}/gene_image{i+1}.png")
        print(f"Image saved: {output_path}/gen_image{i+1}.png")
        
        # Collect frames for video creation (optional)
        frames.append(final_image)
    
    # Save video with each frame displayed for 3-5 seconds
    if num_frames > 1:
        save_video(frames, output_path, fps=0.2)  # 0.2 FPS means 5 seconds per image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to the object image")
    parser.add_argument("--text-prompt", type=str, required=True, help="Text prompt for generating scene")
    parser.add_argument("--output", type=str, required=True, help="Output path for the final images")
    parser.add_argument("--num-images", type=int, default=3, help="Number of scene images to generate")
    parser.add_argument("--num-frames", type=int, default=1, help="Number of frames for video generation")
    
    args = parser.parse_args()
    main(args.image, args.text_prompt, args.output, args.num_images, args.num_frames)
