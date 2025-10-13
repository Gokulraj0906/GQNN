# 🧠 GQNN – Changelog  
_All notable changes to this project will be documented in this file._

## [1.6.0] - 2025-10-13
### Added
- Advanced logging system with automatic fallback and real-time error tracking.
- Model evaluation tools for benchmarking QNN and classical models.
- Full compatibility with **PyTorch ≥ 2.4** and **Transformers ≥ 4.45**.

### Improved
- Documentation overhaul with quick-starts, architecture diagrams, and examples.
- Code stability across multi-GPU and CPU environments.

###  Fixed
- Minor bugs in model saving/loading and tensor device mapping.
- Edge-case issues during distributed training initialization.

---

## [1.5.0] - 2025-09-10
###  Added
- **Hybrid QNN Engine** enabling seamless interaction between quantum and classical neural layers.
- Dataset loader supporting JSONL and CSV domain datasets.
- Error recovery system with retry logic during model setup and inference.

### Improved
- GPU/CPU auto-detection and gradient checkpointing.
- Memory efficiency during fine-tuning and inference.

---

## [1.4.0] - 2025-08-20
### Added
- QLoRA-based fine-tuning pipeline with ready-to-use example notebooks.
- Large-scale domain document ingestion (`main_doc.jsonl` support).
- Live training progress bar and performance summary tracking.

### Improved
- VRAM efficiency in LoRA adapter loading.
- Smoother data loading pipeline for large datasets.

---

## [1.3.0] - 2025-07-18
### Added
- `gqnn.models` module for efficient Quantum Neural Network computation.
- Unified logging system across all components.

### Improved
- Enhanced error handling for missing dependencies or device mismatch.
- Refactored training and inference for modular extensibility.

---

## [1.2.0] - 2025-07-01
### Added
- **Config System** with YAML support for reproducible experiments.
- **Testing Suite** for `fit`, `predict`, and `evaluate` functions.

### Improved
- Import handling and dependency management.

---

## [1.1.0] - 2025-06-10
### Added
- Parameterized quantum circuit layers integrated with PyTorch.
- Simulator backend with Qiskit and PennyLane support.
- Visualization tools for loss and accuracy tracking.

### Improved
- `fit()` method now supports mixed precision training and adaptive optimizers.

---

## [1.0.1] - 2025-02-27
### Updated
- Clarified project scope to include both Quantum and Classical ML support.
- Added structured and practical example code.
- Enhanced Markdown formatting and feature list documentation.

---

## [0.1.1] - 2025-01-13
### Fixed
- Bugs in the `predict` method.
- Missing `@staticmethod` decorator in `_callback_graph`.

### Improved
- `fit()` method handles missing weights gracefully.
- Updated `utils` module functionality.

---

## [0.1.0] - 2025-01-10
### Initial Release
- Core GQNN framework with foundational features.

---

**Note:**  
From **v1.6.0** onward, GQNN officially supports Quantum + Classical hybrid pipelines, domain-specific 

🔮 *Upcoming (v1.7.0)* — Retrieval-Augmented Generation (RAG) support and Quantum-Context hybrid reasoning.