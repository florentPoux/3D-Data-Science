"""
📖 O'Reilly Book: 3D Data Science with Python
=============================================

Code Solution for Chapter: Chapter 02E

General Information:
-------------------
* 🦊 Created by:    Florent Poux
* 📅 Last Update:   Dec. 2025
* © Copyright:      Florent Poux
* 📜 License:       MIT

Dependencies:
------------
* Environment:      Anaconda or Miniconda
* Python Version:   3.9+
* Key Libraries:    NumPy, Pandas, Open3D, Laspy, Scikit-Learn

Helpful Links:
-------------
* 🏠 Author Website:        https://learngeodata.eu
* 📚 O'Reilly Book Page:    https://www.oreilly.com/library/view/3d-data-science/9781098161323/

Enjoy this code! 🚀
"""

import sys
import numpy
import pandas
import matplotlib
import open3d
import laspy
import sklearn

def check_environment():
    """Check if all required libraries are installed and print their versions."""
    print("Environment Verification Script")
    print("-" * 30)
    print(f"Python Version: {sys.version}")
    print("-" * 30)
    print(f"Numpy: {numpy.__version__}")
    print(f"Pandas: {pandas.__version__}")
    print(f"Matplotlib: {matplotlib.__version__}")
    print(f"Open3D: {open3d.__version__}")
    print(f"Laspy: {laspy.__version__}")
    print(f"Scikit-Learn: {sklearn.__version__}")
    print("-" * 30)
    print("Environment setup seems correct!")

if __name__ == "__main__":
    check_environment()