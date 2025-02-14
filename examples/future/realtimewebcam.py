import cv2
import time
from PIL import Image
import os
import numpy as np

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def plot_ascii(image: Image.Image, width: int = 120, color: bool = True) -> list:
    """Convert an image to ASCII art.
    
    Args:
        image: PIL Image to convert
        width: Width of the ASCII art in characters
        color: Whether to use ANSI color codes
        
    Returns:
        List of strings representing the ASCII art
    """
    # Calculate height to maintain aspect ratio
    aspect_ratio = image.height / image.width
    height = int(width * aspect_ratio * 0.5)  # * 0.5 because characters are taller than wide
    
    # Resize image
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    
    # Convert to numpy array for easier processing
    pixels = np.array(image)
    
    # ASCII characters from darkest to lightest
    ascii_chars = '@%#*+=-:. '
    
    # Convert to grayscale for character selection
    if len(pixels.shape) == 3:
        grayscale = np.mean(pixels, axis=2)
    else:
        grayscale = pixels
    
    # Normalize and convert to ASCII
    normalized = (grayscale - grayscale.min()) * (len(ascii_chars) - 1) / (grayscale.max() - grayscale.min())
    
    # Generate ASCII art
    ascii_art = []
    for y in range(height):
        line = []
        for x in range(width):
            char = ascii_chars[int(normalized[y, x])]
            if color and len(pixels.shape) == 3:
                # Get RGB values
                r, g, b = pixels[y, x]
                # Create ANSI color code
                line.append(f"\033[38;2;{r};{g};{b}m{char}\033[0m")
            else:
                line.append(char)
        ascii_art.append(''.join(line))
    
    return ascii_art

def main():
    print("Press Ctrl+C to stop the program.")
    cap = cv2.VideoCapture(0)  # Change to 0 for default camera
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image from webcam.")
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            
            # Convert to ASCII art and display
            ascii_image = plot_ascii(frame, width=120, color=True)
            clear_console()
            print("\n".join(ascii_image))
            
            # Add a small delay to control frame rate
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
    finally:
        cap.release()

if __name__ == "__main__":
    main()
