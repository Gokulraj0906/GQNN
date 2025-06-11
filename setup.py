from setuptools import setup, find_packages

setup(
    name='GQNN',
    version='1.5.0',
    author='GokulRaj S',
    author_email='gokulsenthil0906@gmail.com', 
    description=(
        'QNN is a Python package for Quantum Neural Networks, '
        'a hybrid model combining Quantum Computing and Neural Networks. '
        'It was developed by GokulRaj S for research on Customized Quantum Neural Networks.'
    ),
    long_description=open('README.md',encoding='utf-8').read(), 
    long_description_content_type='text/markdown',
    changelog_path = open('CHANGELOG.md',encoding='utf-8').read(),
    changelog_content_type='text/markdown',
    url='https://github.com/gokulraj0906/GQNN',
    license=open('LICENSE',encoding='utf-8').read(),  
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'qiskit',
        'qiskit_ibm_runtime',
        'qiskit-machine-learning',
        'pylatexenc',
        'matplotlib',
        'torch',
        'tqdm',
        
    ],
     extras_require={
        'linux': ['fireducks']
    },
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
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', 
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7, <=3.12',
    project_urls={
        'Documentation': 'https://www.gokulraj.tech/GQNN/docs',
        'Source': 'https://www.gokulraj.tech/GQNN',
    },
)