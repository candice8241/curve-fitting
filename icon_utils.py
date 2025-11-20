# -*- coding: utf-8 -*-
"""
Icon Utility for XRD Peak Fitting Tool
Provides functions to set window icons with fallback support
"""

import tkinter as tk
from tkinter import PhotoImage
import os
import sys


def set_window_icon(window, icon_path=None, use_default=True):
    """
    Set window icon with multiple fallback options

    Parameters:
    -----------
    window : tk.Tk or tk.Toplevel
        The window to set icon for
    icon_path : str, optional
        Path to custom icon file (.ico, .png, .gif)
    use_default : bool
        Whether to use default icon if custom icon not found

    Returns:
    --------
    success : bool
        True if icon was set successfully
    """

    # Try custom icon path first
    if icon_path and os.path.exists(icon_path):
        try:
            if icon_path.endswith('.ico'):
                # Windows .ico file
                window.iconbitmap(icon_path)
                return True
            else:
                # PNG, GIF, etc.
                icon_image = PhotoImage(file=icon_path)
                window.iconphoto(True, icon_image)
                # Keep a reference to prevent garbage collection
                window._icon_image = icon_image
                return True
        except Exception as e:
            print(f"Failed to load custom icon: {e}")

    # Try default icon locations
    if use_default:
        default_locations = [
            'xrd_icon.ico',
            'xrd_icon.png',
            'icon.ico',
            'icon.png',
            os.path.join(os.path.dirname(__file__), 'xrd_icon.ico'),
            os.path.join(os.path.dirname(__file__), 'xrd_icon.png'),
            os.path.join(os.path.dirname(__file__), 'icons', 'xrd_icon.ico'),
            os.path.join(os.path.dirname(__file__), 'icons', 'xrd_icon.png'),
        ]

        for icon_file in default_locations:
            if os.path.exists(icon_file):
                try:
                    if icon_file.endswith('.ico'):
                        window.iconbitmap(icon_file)
                        return True
                    else:
                        icon_image = PhotoImage(file=icon_file)
                        window.iconphoto(True, icon_image)
                        window._icon_image = icon_image
                        return True
                except Exception as e:
                    continue

    # Try embedded default icon (base64 encoded small PNG)
    try:
        # A simple XRD-themed icon (purple crystal/diffraction pattern)
        # This is a 32x32 purple icon
        default_icon_data = """
        iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABHNCSVQICAgIfAhkiAAAAAlwSFlz
        AAAOxAAADsQBlSsOGwAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAITSURB
        VFiF7ZZNaBNBFMd/s5vNJpuPJk1TtVqkRRBBqIIH8eBBELx48+RJvHjw4EEQvHjw4kUQBA+ePHgQ
        xIMgeFGEYqVQaRU/UGO1abKb3c3uzrwZD7FN3E2yST148J+GzXvz/u/N7JsZYRiGYRiGYRiGYfwv
        EUJM/a8XBAhgfHx8m67rO4QQ24QQW4UQWwEA0HWd8fFxAoGAqqoqiqLgdDrp7e2lq6sLRVEIBoM0
        NDSwZs0aVFVFURRUVaWzs5P6+nqampqoqqpCVVUCgQDhcBifz4fH48Hj8RAKhWhvbyccDuPz+fB6
        vXg8HtxuNx6Ph2AwSFtbG+FwmLa2Nrxe72JZ/+l2u2lpacHv9xMIBKivr6elpQWPx0MgEKC+vh6v
        10s4HKaxsZH6+noaGxtpa2vD5/PR1NSEz+cjFArR0tKC3+9HVVXq6upQFIWmpiYaGhpobm4mEAjg
        9/tRVZVwOExjYyN+vx+fz0dTUxNerxev10tzczM+nw+/309bWxt+v5/m5mZaWlpQFIXm5mZ8Ph+h
        UAiv10tra2vBgJ6eHrq7u+nu7qa7u5uenh56enro7e2lp6eH7u5uent76e7upqenh97eXvr6+ujp
        6aGvr4++vj76+vro7++nv7+fvr4++vv76e/vZ2BggIGBAQYHBxkcHGRwcJChoSGGhoYYGhpiaGiI
        4eFhRkZGGBkZYWRkhNHRUcbGxhgbG2N8fJzJyUkmJycZHx9nYmKCyclJJicn+QNjU41qvE3DVAAA
        AABJRU5ErkJggg==
        """

        import base64
        from io import BytesIO
        from PIL import Image, ImageTk

        # Decode base64 image
        icon_data = base64.b64decode(default_icon_data.strip())
        icon_pil = Image.open(BytesIO(icon_data))
        icon_image = ImageTk.PhotoImage(icon_pil)
        window.iconphoto(True, icon_image)
        window._icon_image = icon_image
        return True

    except Exception as e:
        print(f"Failed to set embedded icon: {e}")
        pass

    return False


def create_simple_xrd_icon(output_path='xrd_icon.png', size=64):
    """
    Create a simple XRD-themed icon using PIL

    Parameters:
    -----------
    output_path : str
        Path to save the icon
    size : int
        Icon size (square)
    """
    try:
        from PIL import Image, ImageDraw, ImageFont

        # Create image with purple gradient background
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Background circle with gradient effect
        for i in range(size//2, 0, -1):
            alpha = int(255 * (i / (size/2)))
            color = (186, 85, 211, alpha)  # Purple with varying alpha
            draw.ellipse([size//2-i, size//2-i, size//2+i, size//2+i],
                        fill=color, outline=None)

        # Draw diffraction pattern (circles representing diffraction rings)
        for radius in [size//6, size//4, size//3]:
            draw.ellipse([size//2-radius, size//2-radius,
                         size//2+radius, size//2+radius],
                        outline=(255, 255, 255, 200), width=2)

        # Draw cross for beam center
        center = size // 2
        cross_size = size // 8
        draw.line([(center-cross_size, center), (center+cross_size, center)],
                 fill=(255, 215, 0, 255), width=3)
        draw.line([(center, center-cross_size), (center, center+cross_size)],
                 fill=(255, 215, 0, 255), width=3)

        # Try to add text
        try:
            font = ImageFont.truetype("arial.ttf", size//4)
        except:
            font = ImageFont.load_default()

        # Add "XRD" text at bottom
        text = "XRD"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = (size - text_width) // 2
        text_y = size - text_height - size//10

        # Text shadow
        draw.text((text_x+1, text_y+1), text, fill=(0, 0, 0, 200), font=font)
        # Main text
        draw.text((text_x, text_y), text, fill=(255, 255, 255, 255), font=font)

        # Save the image
        img.save(output_path)
        print(f"Icon created successfully: {output_path}")

        # Also try to create .ico version for Windows
        if output_path.endswith('.png'):
            ico_path = output_path.replace('.png', '.ico')
            try:
                img.save(ico_path, format='ICO', sizes=[(size, size)])
                print(f"ICO icon created: {ico_path}")
            except Exception as e:
                print(f"Could not create ICO file: {e}")

        return True

    except ImportError:
        print("PIL/Pillow not installed. Install with: pip install Pillow")
        return False
    except Exception as e:
        print(f"Failed to create icon: {e}")
        return False


def generate_icon_variations():
    """Generate icon in multiple sizes for different uses"""
    sizes = [16, 32, 48, 64, 128, 256]

    for size in sizes:
        output_path = f'xrd_icon_{size}.png'
        create_simple_xrd_icon(output_path, size)

    # Create the default icon
    create_simple_xrd_icon('xrd_icon.png', 64)
    create_simple_xrd_icon('xrd_icon.ico', 64)

    print("\nIcon generation complete!")
    print("Available icons:")
    for size in sizes:
        print(f"  - xrd_icon_{size}.png")
    print("  - xrd_icon.png (default, 64x64)")
    print("  - xrd_icon.ico (Windows icon)")


if __name__ == "__main__":
    print("XRD Icon Generator")
    print("=" * 50)

    import argparse
    parser = argparse.ArgumentParser(description='Generate XRD icons')
    parser.add_argument('--size', type=int, default=64,
                       help='Icon size (default: 64)')
    parser.add_argument('--output', type=str, default='xrd_icon.png',
                       help='Output path (default: xrd_icon.png)')
    parser.add_argument('--all', action='store_true',
                       help='Generate all size variations')

    args = parser.parse_args()

    if args.all:
        generate_icon_variations()
    else:
        create_simple_xrd_icon(args.output, args.size)
