# GraphQuery-LLM 🚀

> **Intelligent Graph-Based Question Answering System powered by Neo4j, APOC, and OpenAI GPT-4**

GraphQuery-LLM is a high-performance Graph Retrieval-Augmented Generation (Graph RAG) system that combines the power of knowledge graphs with large language models to provide accurate, context-aware answers to complex questions.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.0+-green.svg)](https://neo4j.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange.svg)](https://openai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 Features

- **🧠 Intelligent Entity Extraction**: Automatically identifies relevant entities from natural language questions using LLM
- **🔍 Fuzzy Node Matching**: Finds matching nodes in Neo4j graph using exact and fuzzy search strategies
- **🕸️ Multi-hop Graph Traversal**: Leverages APOC procedures for efficient subgraph exploration
- **⚡ Parallel Processing**: ThreadPoolExecutor for concurrent subgraph queries (up to 3 workers)
- **💾 Smart Caching**: In-memory caching for schemas, node matches, and subgraphs to reduce redundant queries
- **📊 Detailed Performance Metrics**: Track timing for each pipeline stage with comprehensive breakdowns
- **🎯 Context-Aware Answers**: Generates answers based on actual graph relationships with path visualization

---

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Usage Examples](#-usage-examples)
- [Performance Metrics](#-performance-metrics)
- [System Architecture](#-system-architecture)
- [API Reference](#-api-reference)
- [Optimization Tips](#-optimization-tips)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🏗️ Architecture

GraphQuery-LLM processes questions through a 3-phase pipeline:
