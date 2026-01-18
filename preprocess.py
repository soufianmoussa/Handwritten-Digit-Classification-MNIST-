import numpy as np
from PIL import Image, ImageOps

def preprocess_image(image_pil):
    """
    Preprocesses a PIL image for MNIST digit classification (28x28).
    
    Steps:
    1. Convert to grayscale.
    2. Invert colors (ensure white digit on black background).
    3. Crop to bounding box (remove empty borders).
    4. Resize while preserving aspect ratio to fit in a 20x20 box.
    5. Pad to 28x28 (centering the digit by center of mass logic).
    6. Normalize and flat.
    """
    # 1. Convert to grayscale
    img = image_pil.convert('L')
    
    # 2. Invert colors (Canvas is usually black drawing on white background)
    #    We assume the input is black-digit-on-white.
    if np.mean(np.array(img)) > 127:
        img = ImageOps.invert(img)
    
    # 3. Crop to bounding box
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
        
    # 4. Resize to fit in 20x20 box (leaving 4px padding on all sides for 28x28)
    #    This mimics MNIST construction where the digit is in a 20x20 box centered in 28x28 field.
    w, h = img.size
    
    # Max dimension should be 20
    max_dim = 20
    ratio = max_dim / max(h, w)
    new_size = (int(w * ratio), int(h * ratio))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # 5. Pad to 28x28 (Center it)
    new_img = Image.new('L', (28, 28), 0)
    
    # Paste centered
    offset_x = (28 - new_size[0]) // 2
    offset_y = (28 - new_size[1]) // 2
    new_img.paste(img, (offset_x, offset_y))
    
    # 6. Normalize
    img_array = np.array(new_img)
    img_array = img_array / 255.0
    
    # Flatten (1, 784)
    # MNIST data is usually row-major [0,0], [0,1]... 
    # Our previous code used 'F' order because of MATLAB legacy.
    # Standard MNIST loaders (including the one I wrote) usually return C-contiguous (row-major).
    # so we will use default order='C'.
    return img_array.flatten().reshape(1, 784), img_array
