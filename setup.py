import setuptools
import os

_PROJECT_NAME = "AetherManipulator"
_PROJECT_VERSION = "0.1.0"  # Start with an initial version
_PROJECT_AUTHOR = "Omdeep Borkar"
_PROJECT_AUTHOR_EMAIL = "omdeeborkar@gmail.com"
_PROJECT_URL = "https://github.com/Omdeepb69/AetherManipulator"
_PROJECT_DESCRIPTION = (
    "An interactive Python application using MediaPipe to track hand gestures "
    "in real-time, allowing users to manipulate 3D models on screen as if "
    "handling holographic projections, you know, for science!"
)
_PROJECT_CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License", # Assuming MIT, change if needed
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Visualization",
    "Topic :: Multimedia :: Graphics :: 3D Modeling",
    "Topic :: Software Development :: Libraries :: Python Modules",
]
_PYTHON_REQUIRES = ">=3.8"
_INSTALL_REQUIRES = [
    "opencv-python>=4.5",
    "mediapipe>=0.9",
    "numpy>=1.20",
    "PyOpenGL>=3.1",
    "PyOpenGL_accelerate>=3.1", # Often needed with PyOpenGL
    "pywavefront>=1.3",
]

# Try to read the README file for the long description
try:
    _LONG_DESCRIPTION = open("README.md", encoding="utf-8").read()
except FileNotFoundError:
    _LONG_DESCRIPTION = _PROJECT_DESCRIPTION


setuptools.setup(
    name=_PROJECT_NAME,
    version=_PROJECT_VERSION,
    author=_PROJECT_AUTHOR,
    author_email=_PROJECT_AUTHOR_EMAIL,
    description=_PROJECT_DESCRIPTION,
    long_description=_LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=_PROJECT_URL,
    packages=setuptools.find_packages(where="src"), # Assumes code is in 'src' dir
    package_dir={"": "src"}, # Tells setuptools packages are under src
    include_package_data=True, # Include non-code files specified in MANIFEST.in
    classifiers=_PROJECT_CLASSIFIERS,
    python_requires=_PYTHON_REQUIRES,
    install_requires=_INSTALL_REQUIRES,
    # Add entry points if you have command-line scripts
    # entry_points={
    #     'console_scripts': [
    #         'aethermanipulator=aethermanipulator.main:run',
    #     ],
    # },
    project_urls={
        "Bug Tracker": f"{_PROJECT_URL}/issues",
        "Source Code": _PROJECT_URL,
    },
    keywords="mediapipe hand tracking gesture control 3d opengl visualization",
)