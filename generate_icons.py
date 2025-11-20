#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速图标生成脚本
一键生成XRD峰拟合工具所需的所有图标
"""

import os
import sys

def main():
    print("="*60)
    print("XRD Peak Fitting Tool - Icon Generator")
    print("="*60)
    print()

    # Check if Pillow is installed
    try:
        from PIL import Image
        print("✓ PIL/Pillow is installed")
    except ImportError:
        print("✗ PIL/Pillow is not installed!")
        print()
        print("Please install it using:")
        print("  pip install Pillow")
        print()
        sys.exit(1)

    # Import icon utilities
    try:
        from icon_utils import generate_icon_variations, create_simple_xrd_icon
        print("✓ Icon utilities loaded successfully")
    except ImportError as e:
        print(f"✗ Failed to import icon_utils: {e}")
        sys.exit(1)

    print()
    print("Generating icons...")
    print("-" * 60)

    # Generate all icon variations
    generate_icon_variations()

    print()
    print("="*60)
    print("✅ Icon Generation Complete!")
    print("="*60)
    print()
    print("Generated files:")
    print("  • xrd_icon.png       - Default icon (64x64)")
    print("  • xrd_icon.ico       - Windows icon")
    print("  • xrd_icon_16.png    - Small icon")
    print("  • xrd_icon_32.png    - Medium icon")
    print("  • xrd_icon_48.png    - Large icon")
    print("  • xrd_icon_64.png    - Extra large icon")
    print("  • xrd_icon_128.png   - High resolution")
    print("  • xrd_icon_256.png   - Very high resolution")
    print()
    print("Next steps:")
    print("  1. Icons are ready to use!")
    print("  2. Run your application:")
    print("     python peak_fitting_gui_enhanced.py")
    print("  3. Or run the integrated module:")
    print("     python test_integrated_module.py")
    print()
    print("The icons will be automatically loaded by the application.")
    print("="*60)


if __name__ == "__main__":
    main()
