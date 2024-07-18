# LocalPubMed-RAG'n'Roll

## Overview

LocalPubMed-RAGto Riches is a solution that integrates PubMed’s vast database of biomedical abstracts with an efficient local retrieval system. This setup uses advanced techniques like binary quantization, HDF5-based vectorization bunged in with the Bio Mistral-Instruct 7B model to perform high-speed, memory-efficient searches on a local system. The goal is to enable  natural language queries without the need for internet access, ensuring data security and privacy.

## Features

- **Binary Quantization with Qdrant**: Achieves up to 40x retrieval speedup and reduces memory consumption.
- **HDF5-based Vectorization**: Supports parallel I/O operations for high-performance computing.
- **Bio Mistral-Instruct 7B Model**: Fine-tuned on the PubMedQA dataset for efficient and relevant query processing.
- **Local Execution**: Ensures data privacy and security by running entirely on a local system.

## Installation

### Prerequisites

- Docker
- Python 3.8+
- Git

### Setup

1. **Clone the repository**:
    ```bash
    git clone https://github.com/kirandas-dev/LocalPubMed-RAGtoRiches.git
    cd LocalPubMed-RAGtoRiches
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Pull Qdrant Docker Image**:
    ```bash
    docker pull qdrant/qdrant
    ```

4. **Run Qdrant**:
    ```bash
    docker run -p 6333:6333 qdrant/qdrant
    ```

5. **Download PubMed Data**:
    - Get the PubMed SQLite database from the official source.
    - Convert the database to HDF5 format.

## Usage

1. **Vectorization**:
    - Convert the abstracts to vectors using HDF5 and apply binary quantization.
    - Example script:
        ```python
        python vectorize_pubmed.py --input pubmed.sqlite --output pubmed.h5
        ```

2. **Model Integration**:
    - Use the `llama.cpp` repo to convert the model to GGUF format with Metal support for hardware acceleration.
    - Example script:
        ```bash
        git clone https://github.com/ggerganov/llama.cpp.git
        cd llama.cpp
        make
        ./main -m models/gguf --n_gpu_layers 1
        ```

3. **Running Queries**:
    - Use Langchain to integrate the model with the RAG pipeline.
  

## Evaluation

### Performance Metrics

- **Load Time**: 4.856 seconds for loading a large model.
- **Sample Time**: 0.01805 seconds for 180 runs.
- **Prompt Evaluation Time**: 21.41833 seconds for 431 tokens.
- **Evaluation Time**: 15.91611 seconds for 179 runs.

### Optimization Tips

- Focus on improving prompt evaluation times to reduce latency.
- Experiment with different `n_gpu_layers` settings to balance GPU and CPU usage effectively.

## Future Work

- Further fine-tuning of model parameters and prompt engineering.
- Exploration of additional vectorization techniques and storage formats.
- Continuous integration and deployment enhancements.

## Contributing

Feel free to fork this repository, submit issues, and send pull requests. Contributions are welcome!

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- PubMed for providing the biomedical abstracts database.
- Qdrant and Langchain for their powerful tools and libraries.
- The open-source community for their invaluable contributions.
