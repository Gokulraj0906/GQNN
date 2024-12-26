from setuptools import setup, find_packages

setup(
    name='QNN',
    version='0.1.0',
    author='GokulRaj S',
    author_email='gokulsenthil0906@gmail.com', 
    description=(
        'QNN is a Python package for Quantum Neural Networks, '
        'a hybrid model combining Quantum Computing and Neural Networks. '
        'It was developed by GokulRaj S for research on Customized Quantum Neural Networks.'
    ),
    long_description=open('README.md').read(), 
    long_description_content_type='text/markdown',
    url='https://github.com/gokulraj0906/QNN',  
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'fireducks',
        'scikit-learn',
        'qiskit'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', 
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.9', 
)
