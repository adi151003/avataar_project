# avataar_project
Place an object’s image in a text-conditioned scene

This code focuses on generating realistic scenes by combining various techniques such as scene segmentation, object detection, and lighting adjustments to place objects logically within generated environments. The main goal is to integrate an object image into a generated scene based on a text prompt, ensuring its placement is contextually appropriate and visually convincing.
Scene Segmentation: The code uses a pre-trained DeepLabV3 model for scene segmentation, identifying regions within the scene image (like tables or counters) suitable for placing objects. This segmentation helps restrict placements to logical areas by analyzing the pixel-wise layout of the scene.
Object Detection: YOLOv8 is employed to detect existing objects within the scene to avoid overlaps. After segmentation, valid positions are filtered by ensuring they do not interfere with detected bounding boxes from YOLOv8. This step avoids unrealistic overlap between the newly placed object and pre-existing elements in the scene.
Object Scaling and Lighting Adjustment: Objects are resized proportionally to fit within 15-25% of the scene dimensions, ensuring they don’t dominate the image. After scaling, the object’s brightness is adjusted using ImageEnhance.Brightness to match the lighting conditions of the scene, enhancing realism.
Placement Strategy: The valid positions from segmentation are narrowed down further by ensuring no overlap with detected objects, and a final valid position is chosen randomly. If no valid position is found, a default random placement within the scene bounds is used.
The code also integrates scene generation via a Stable Diffusion model, generating images based on the text prompt.

Failed approaches involved simpler object placement logic without segmentation or detection, which led to unrealistic images with poor lighting, size mismatches, and illogical object
Previous attempts focused solely on basic object segmentation or bounding boxes for placement, often failing to produce believable results. This lack of context-sensitive placement led to unrealistic scenes, as objects were either too large, poorly lit, or incorrectly positioned, reducing realism.



![gene_image1](https://github.com/user-attachments/assets/4c47b8c1-9458-42b7-87ab-9ff3f1e67399)
![gene_image9](https://github.com/user-attachments/assets/bc0b96e7-c12c-4e6b-9e7d-7298594c2a88)
![gene_image8](https://github.com/user-attachments/assets/61b053a5-a2d3-4041-8094-41f1547e3035)
![gene_image7](https://github.com/user-attachments/assets/4f5e81bb-fcdf-4c21-bdba-c2e035d6f475)
![gene_image6](https://github.com/user-attachments/assets/882bbb58-1b15-4b51-aac8-7d57a5436023)
![gene_image5](https://github.com/user-attachments/assets/103be8ac-7dc9-4fc5-8819-6f690c5cee5c)
![gene_image4](https://github.com/user-attachments/assets/9f1f5325-f976-410d-b6ab-6d4f7b56a52e)
![gene_image3](https://github.com/user-attachments/assets/1711aea7-0c98-4ceb-9456-61e7ebd6889f)
![gene_image2](https://github.com/user-attachments/assets/dd6e4122-dcbe-4476-9e68-4f196d9f39a6)
![gene_image10](https://github.com/user-attachments/assets/a3ae1ff6-e873-49ee-ae41-9602b3e4fec2)


https://github.com/user-attachments/assets/80f42ba1-8c54-4855-b407-7a65bbe3f63c

