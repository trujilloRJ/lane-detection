import cv2
import os

def create_video_from_images(image_folder, output_video, frame_rate=30, quality=90):
    images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()  # Ensure the images are in the correct order
    
    if not images:
        print("No images found in the specified folder.")
        return
    
    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = first_image.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))
    
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)

        # Compress the image before writing to video
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded_image = cv2.imencode('.jpg', frame, encode_param)
        frame = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)

        video.write(frame)
    
    video.release()
    print(f"Video saved as {output_video}")

# Example usage
image_folder = f"examples/BUnet_d4_c32_a2_SOneCycle"  
output_video = 'output_video.mp4'
fps = 2
create_video_from_images(image_folder, output_video, frame_rate=fps)