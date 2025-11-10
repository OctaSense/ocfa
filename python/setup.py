"""
OCTA Face SDK - Python Package Setup
"""

from setuptools import setup, find_packages
import os

# Read README
def read_file(filename):
    with open(os.path.join(os.path.dirname(__file__), '..', filename), 'r', encoding='utf-8') as f:
        return f.read()

setup(
    name='ocfa-face-sdk',
    version='1.0.0',
    author='OCTA Team',
    author_email='support@your-company.com',
    description='Embedded Face Recognition SDK with RGB-IR Liveness Detection',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/your-repo/octa-face',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.10.0',
        'onnx>=1.12.0',
        'onnxruntime>=1.12.0',
        'opencv-python>=4.6.0',
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'scikit-learn>=1.0.0',
        'Pillow>=9.0.0',
        'tqdm>=4.62.0',
        'pyyaml>=6.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'flake8>=4.0.0',
            'black>=22.0.0',
            'mypy>=0.950',
        ],
        'docs': [
            'sphinx>=4.5.0',
            'sphinx-rtd-theme>=1.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'ocfa-export=tools.model_export:main',
            'ocfa-quantize=tools.quantization:main',
            'ocfa-benchmark=tools.benchmark:main',
            'ocfa-evaluate=tools.evaluate:main',
        ],
    },
)
