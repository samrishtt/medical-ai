from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="derm-equity",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Equitable Skin Cancer Detection via Uncertainty-Aware Multi-Scale Vision Transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/derm-equity",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/derm-equity/issues",
        "Documentation": "https://github.com/yourusername/derm-equity#readme",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pytorch-lightning>=2.0.0",
        "timm>=0.9.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "albumentations>=1.3.0",
        "Pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "omegaconf>=2.3.0",
        "wandb>=0.15.0",
        "gradio>=4.0.0",
        "tqdm>=4.65.0",
        "einops>=0.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "medical": [
            "pydicom>=2.4.0",
            "nibabel>=5.0.0",
            "SimpleITK>=2.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "derm-train=scripts.train:main",
            "derm-eval=scripts.evaluate:main",
            "derm-demo=demo.app:main",
        ],
    },
)
