from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Optional: Load license and changelog safely
try:
    with open("LICENSE", "r", encoding="utf-8") as f:
        license_text = f.read()
except FileNotFoundError:
    license_text = ""

try:
    with open("CHANGELOG.md", "r", encoding="utf-8") as f:  # Use .md extension
        changelog_text = f.read()
except FileNotFoundError:
    changelog_text = ""

setup(
    name='GQNN',
    version='1.0.1',
    author='GokulRaj S',
    author_email='gokulsenthil0906@gmail.com',
    description=(
        'QNN is a Python package for Quantum Neural Networks, '
        'a hybrid model combining Quantum Computing and Neural Networks. '
        'It was developed by GokulRaj S for research on Customized Quantum Neural Networks.'
    ),
    long_description=long_description + "\n\n" + changelog_text,  # Append CHANGELOG to long_description
    long_description_content_type='text/markdown',
    url='https://github.com/gokulraj0906/GQNN',
    license=license_text,  # Safe handling of LICENSE file
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'qiskit',
        'qiskit_ibm_runtime',
        'qiskit-machine-learning',
        'pylatexenc',
        'ipython',
        'matplotlib',
    ],
    extras_require={
        'linux': ['fireducks']
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7',
    
    # ✅ Added Keywords to Improve PyPI Search Visibility
    keywords=[
        "quantum computing",
        "quantum neural networks",
        "QNN",
        "machine learning",
        "artificial intelligence",
        "quantum machine learning",
        "deep learning",
        "qiskit",
        "quantum algorithms",
        "scientific computing",
        "research",
        "data science",
        "quantum AI",
        "quantum optimization",
    ],
)
