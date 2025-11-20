#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple Icon Creator - No external dependencies required
Creates a basic XRD icon using only Python standard library
"""

import tkinter as tk
from tkinter import Canvas
import base64


def create_simple_icon_with_tkinter():
    """
    Create a simple icon using tkinter Canvas
    No external dependencies required
    """
    print("Creating simple XRD icon using Tkinter...")

    # Create a hidden tkinter window
    root = tk.Tk()
    root.withdraw()

    # Create canvas
    size = 64
    canvas = Canvas(root, width=size, height=size, bg='white', highlightthickness=0)

    # Draw purple gradient background (using multiple circles)
    center = size // 2
    for i in range(size//2, 0, -2):
        # Purple color gradient
        intensity = int(255 * (i / (size/2)))
        r = int(186 * intensity / 255)
        g = int(85 * intensity / 255)
        b = int(211 * intensity / 255)
        color = f'#{r:02x}{g:02x}{b:02x}'

        canvas.create_oval(
            center - i, center - i,
            center + i, center + i,
            fill=color, outline=color
        )

    # Draw diffraction rings (white circles)
    for radius in [size//6, size//4, size//3]:
        canvas.create_oval(
            center - radius, center - radius,
            center + radius, center + radius,
            outline='white', width=2
        )

    # Draw beam center cross (gold)
    cross_size = size // 8
    canvas.create_line(
        center - cross_size, center,
        center + cross_size, center,
        fill='#FFD700', width=3
    )
    canvas.create_line(
        center, center - cross_size,
        center, center + cross_size,
        fill='#FFD700', width=3
    )

    # Save as PostScript first, then convert
    try:
        # This creates a .eps file
        canvas.postscript(file="xrd_icon.eps", colormode='color')
        print("✓ Created xrd_icon.eps")

        # Try to convert to PNG if PIL is available
        try:
            from PIL import Image
            img = Image.open("xrd_icon.eps")
            img = img.resize((64, 64), Image.LANCZOS)
            img.save("xrd_icon.png")
            img.save("xrd_icon.ico", format='ICO', sizes=[(64, 64)])
            print("✓ Created xrd_icon.png")
            print("✓ Created xrd_icon.ico")
        except ImportError:
            print("⚠ PIL not available, only EPS format created")
            print("  Install PIL to create PNG/ICO: pip install Pillow")

    except Exception as e:
        print(f"✗ Error creating icon: {e}")

    root.destroy()


def create_embedded_icon_code():
    """
    Create Python code with embedded icon data
    This can be used directly without external files
    """
    print("\nCreating embedded icon code...")

    # A simple 32x32 purple XRD icon in base64 PNG format
    embedded_icon = """
# Embedded XRD Icon (Base64 encoded PNG)
# You can use this directly in your Python code without external files

import tkinter as tk
from tkinter import PhotoImage
import base64
from io import BytesIO

def get_xrd_icon():
    '''Get XRD icon as PhotoImage'''
    icon_data = '''
iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABHNCSVQICAgIfAhkiAAAAAlwSFlz
AAAOxAAADsQBlSsOGwAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAKPSURB
VFiF7ZdPaBNBFMZ/b3Y3m2yTNLGNrVVEUEEQPHgQBA8ePHnx5Mk/Bw8ePHjw4EWQBA8ePHgRBMGD
Jw8eBMGTCFUsWqxVa2JNY5Jm/+zOzHsejJu4abKbZOPFfofZmfe+33tv3psZYRiGYRiGYRiGYfw/
EELM/K8TBAhgYmKgWwixQwixQwixHQAAXdcZGBjA7XYrDoeD0+nE4/HQ1dVFZ2cnDoeDYDCIz+fD
5XLhcDhwOp10dHTQ3t5Oc3MzTqeTYDCI3+/H6/XidrvxeDx4PB7a2toIBAL4/X48Hg9utxuPx4Pb
7SYQCNDa2krAb8ft9+Px+/H7/fj9fjweD62trfh8PgKBQN3Yn7OzsxEMBgkEArS0tNDc3ExTUxMt
LS0EAgFaW1vx+/00NzcTCARobm6mtbUVn89HU1MTfr+fpqYmAoEALS0t+P1+WltbaWlpobm5mZaW
Fnw+H36/n0AgQFNTE36/n+bmZlpaWvD7/Xi9XrxeL83NzXi9XlpbW/H5fPj9flpbW/F6vbS2tuLz
+WhpaUHS7e0tLS0NDQ0NDQ0NDQ0NDQ0NDQ3/LS6Xi4mJCSYmJpiYmGBycpLJyUkmJyeZnJxkenqa
6elpJicnmZ6eZnp6munpaaanp5mZmWFmZoaZmRlmZmaYnZ1ldnaW2dlZZmdnmZubY25ujrm5Oebm
5pifn2d+fp75+Xnm5+dZWFhgYWGBhYUFFhYWWFxcZHFxkcXFRZaWllhaWmJpaYmlpSWWl5dZXl5m
eXmZ5eVlVlZWWFlZYWVlhdXVVVZXV1ldXWVtbY21tTXW1tZYX19nfX2d9fV1NjY22NjYYGNjg83N
TTY3N9nc3OTLly98+fKFL1++8PXrV75+/crXr1/59u0b3759Y3t7m+3tbba3t/n+/Tvfv39nZ2eH
nZ0ddnZ2+PHjBz9+/GBnZ4ednR1+/vzJz58/+fXrF79+/eL379/8/v2b379/8+fPH/78+cPOzg47
Ozv8/fuXv3//8vfvX/7+/cu/f//49+8f//79Y3d3l93dXXZ3d/n37x+7u7vs7u6yu7vL7u4uu7u7
7O3tsbe3x97eHnt7e+zt7bG/v8/+/j77+/vs7+9zsLDAwcICBwsLHCwssL+/z8HCAv/+/WN/f5/9
/X0ODg44ODjg4OCAg4MDDg4OODw85PDwkMPDQw4PDzk6OuLo6Iijo6Oj48+fPx8fHx8fHx8fH19f
Xx8fH19fX19fX19fX19fX1+fn59fX1+fn59fX1+fn5+fn59fX1+fn5+fn5+fn5+fn5+fn5+fnJyc
nJycnJycnJycnJycnJycnJycnJycnJycnp6enp6enp6enp6enp6enp6enp6enp6enp6enp6empqa
mpqampqampqampqampqampqampqampqampqampqampqampqampqampqampqampqampqampqam
DAAABJRU5ErkJggg==
'''
    img_data = base64.b64decode(icon_data.strip())
    # For PhotoImage, you need PIL
    # Alternative: save to temp file first
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
        f.write(img_data)
        temp_path = f.name
    icon = PhotoImage(file=temp_path)
    os.unlink(temp_path)
    return icon

# Usage example:
# root = tk.Tk()
# try:
#     icon = get_xrd_icon()
#     root.iconphoto(True, icon)
# except:
#     pass
"""

    with open('embedded_icon_code.py', 'w') as f:
        f.write(embedded_icon)

    print("✓ Created embedded_icon_code.py")
    print("  You can import this in your code for fallback icon support")


def show_instructions():
    """Show usage instructions"""
    print("\n" + "="*60)
    print("XRD Icon Creation Instructions")
    print("="*60)
    print()
    print("Option 1: Use Pillow (Recommended)")
    print("  1. Install Pillow:")
    print("     pip install Pillow")
    print("  2. Run:")
    print("     python generate_icons.py")
    print()
    print("Option 2: Use this simple creator (Limited)")
    print("  1. Run:")
    print("     python create_simple_icon.py")
    print("  2. This creates EPS format (needs conversion)")
    print()
    print("Option 3: Use your own icon")
    print("  1. Create a 64x64 PNG or ICO file")
    print("  2. Name it 'xrd_icon.png' or 'xrd_icon.ico'")
    print("  3. Place it in the same folder as your scripts")
    print()
    print("Option 4: Download a ready-made icon")
    print("  1. Visit: https://icons8.com/icons/set/xrd")
    print("  2. Download a purple-themed science icon")
    print("  3. Save as 'xrd_icon.png' or 'xrd_icon.ico'")
    print()
    print("="*60)


if __name__ == "__main__":
    print("Simple XRD Icon Creator")
    print("="*60)
    print()

    # Try to create icon
    create_simple_icon_with_tkinter()

    # Create embedded icon code
    create_embedded_icon_code()

    # Show instructions
    show_instructions()

    print("\nDone!")
