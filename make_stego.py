"""
make_stego.py — Generate LSB steganography images for demo
──────────────────────────────────────────────────────────
Uses the same LSB embedding method as the diegozanchett dataset
so the model will reliably detect them.

Usage:
    python make_stego.py --input clean.jpg --output stego.jpg
    python make_stego.py --input clean.jpg --output stego.jpg --message "SECRET DATA"
    python make_stego.py --batch --input_dir ./clean_images --output_dir ./stego_images
"""

import argparse
import os
import numpy as np
from PIL import Image


def text_to_bits(text: str) -> list:
    """Convert text to a flat list of bits."""
    bits = []
    for char in text:
        byte = ord(char)
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    # Append 16 zero bits as end marker
    bits.extend([0] * 16)
    return bits


def embed_lsb(image_path: str, output_path: str, message: str = None) -> bool:
    """
    Embed a message into an image using LSB steganography.
    Matches the exact method used in the training dataset.
    
    If no message provided, embeds random bits (still detectable by model).
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)

    if message is None:
        # Embed random bits across ~30% of pixels (matches training data density)
        rng = np.random.RandomState(42)
        n_pixels = arr.shape[0] * arr.shape[1]
        n_embed  = int(n_pixels * 0.30)
        
        # Flatten, embed, reshape
        flat = arr.flatten()
        indices = rng.choice(len(flat), n_embed, replace=False)
        random_bits = rng.randint(0, 2, n_embed).astype(np.uint8)
        
        # Clear LSB and set new bit
        flat[indices] = (flat[indices] & 0xFE) | random_bits
        arr = flat.reshape(arr.shape)
    else:
        # Embed actual message bits
        bits = text_to_bits(message)
        flat = arr.flatten()
        
        if len(bits) > len(flat):
            print(f"Message too long! Max {len(flat)//8} characters for this image.")
            return False
        
        for i, bit in enumerate(bits):
            flat[i] = (flat[i] & 0xFE) | bit
        
        arr = flat.reshape(arr.shape)

    result = Image.fromarray(arr.astype(np.uint8))
    result.save(output_path, format="PNG")  # PNG to avoid JPEG re-compression
    print(f"✅ Stego image saved: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate LSB stego images for demo")
    parser.add_argument("--input",      type=str, help="Input clean image path")
    parser.add_argument("--output",     type=str, help="Output stego image path")
    parser.add_argument("--message",    type=str, default=None,
                        help="Message to hide (default: random bits)")
    parser.add_argument("--batch",      action="store_true",
                        help="Process entire directory")
    parser.add_argument("--input_dir",  type=str, help="Input directory for batch mode")
    parser.add_argument("--output_dir", type=str, help="Output directory for batch mode")
    args = parser.parse_args()

    if args.batch:
        if not args.input_dir or not args.output_dir:
            print("Batch mode requires --input_dir and --output_dir")
            return
        
        os.makedirs(args.output_dir, exist_ok=True)
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(exts)]
        
        print(f"Processing {len(files)} images...")
        for fname in files:
            inp  = os.path.join(args.input_dir, fname)
            base = os.path.splitext(fname)[0]
            out  = os.path.join(args.output_dir, f"{base}_stego.png")
            embed_lsb(inp, out, args.message)
        
        print(f"\n✅ Done! {len(files)} stego images saved to {args.output_dir}")

    elif args.input and args.output:
        msg = args.message or "BARCLAYS_CONFIDENTIAL: Account 4839201 Sort 20-14-55"
        print(f"Embedding message: '{msg}'")
        embed_lsb(args.input, args.output, msg)

    else:
        # Demo mode — create a test stego image from scratch
        print("Demo mode — creating test stego image...")
        
        # Create a simple clean image if no input given
        demo_clean = "demo_clean.png"
        demo_stego = "demo_stego.png"
        
        if not os.path.exists(demo_clean):
            # Create a gradient image as clean test
            arr = np.zeros((256, 256, 3), dtype=np.uint8)
            for i in range(256):
                arr[i, :, 0] = i          # Red gradient
                arr[:, i, 2] = 255 - i    # Blue gradient
                arr[i, :, 1] = 128        # Green constant
            Image.fromarray(arr).save(demo_clean)
            print(f"Created clean test image: {demo_clean}")
        
        msg = "BARCLAYS_CONFIDENTIAL: Customer PAN 4539 1488 0343 6467, Sort: 20-14-55"
        embed_lsb(demo_clean, demo_stego, msg)
        
        print(f"\nTest files created:")
        print(f"  Clean: {demo_clean}  → should show CLEAN in app")
        print(f"  Stego: {demo_stego}  → should show DETECTED in app")
        print(f"\nRun: streamlit run app.py")
        print(f"Then upload both files to compare results.")


if __name__ == "__main__":
    main()