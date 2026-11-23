#!/usr/bin/env python3
"""AIcare 5 - portable intensive care monitor demonstrator with AI decision support."""

import flet as ft

from aicare.app import main

if __name__ == "__main__":
    ft.run(main, assets_dir="assets")
