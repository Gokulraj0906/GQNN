from setuptools import setup, find_packages

setup(
    name='GQNN',
    version='0.1.1',
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
)