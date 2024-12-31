"""
QNN Package

This is the core module for the QNN (Quantum Neural Networks) Python package, 
developed by GokulRaj S. It is designed for hybrid models combining Quantum Computing 
and Neural Networks. The package provides tools for building, training, and evaluating 
customized Quantum Neural Networks.

Package Metadata:
-----------------
- Author: GokulRaj S
- Version: 0.1.0
- License: MIT
- Maintainer: GokulRaj S
- Email: gokulsenthil0906@gmail.com
- Status: Development
- Description: A Python package for Quantum Neural Networks.
- Keywords: Quantum Neural Networks, Quantum Computing, Machine Learning, Neural Networks
- URLs:
    - GitHub: 
    - Documentation: 
    - Bug Report: 
    - Funding: 
    - Contributions: 
    - Citation: 
    - Donation: 
    - Acknowledgement: 
    - Reference: 
    - Community: 
    - Blog: 
    - News: 
    - Publications: 
    - Videos: 
    - Presentations: 
    - Tutorials: 
    - Workshops: 
    - Conferences: 
    - Events: 
    - Meetups: 
    - Webinars: 
    - Podcasts: 
    - Interviews: 
    - Case Studies: 

"""

from QNN.data import dataset
from QNN.models import Linear_model
from QNN.models import save_models
from QNN.models import data_split
from QNN.validation import validation
from QNN.data import rfe
from QNN.data import pca

__all__ = ["dataset","Linear_model","save_models","data_split","rfe","pca"]

__author__ = "GokulRaj S"
__version__ = "0.1.0"
__license__ = "MIT"
__maintainer__ = "GokulRaj S"
__email__ = "gokulsenthil0906@gmail.com"
__status__ = "Development"
__description__ = "A Python package for Quantum Neural Networks"
__keywords__ = "Quantum Neural Networks, Quantum Computing, Machine Learning, Neural Networks"
__url__ = "https://www.gokulraj.tech/QNN"
__github_url__ = "https://github.com/gokulraj0906/QNN"
__documentation_url__ = "https://www.gokulraj.tech/QNN/docs"
__bug_report_url__ = "https://www.gokulraj.tech/QNN/report"
__funding_url__ = "https://www.gokulraj.tech/QNN/support"
__tutorial_url__ = "https://www.gokulraj.tech/QNN/tutorials"
