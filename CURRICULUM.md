# AI Engineering & Data Science Curriculum

The following is a high level curriculum that I am going through in an effort to teach myself about all aspects of AI from data exploration to running a fully fledged LLM in a cloud environment with GPU nodes... let's see how it goes and chances are it will change and evolve as I go through the various labs and come across new and interesting things.

## Module 1 — Foundations of AI Modelling

### Learning Objectives:
- Understand the end-to-end AI lifecycle from data to production
- Be fluent in terminology used by Data Scientists and AI Engineers
- Recognise how models fit into larger distributed systems
- Understand key trade-offs: speed vs accuracy vs cost

### Labs:
| Lab | Focus | Starter Prompt |
|-----|-------|----------------|
| Lab 1 | AI Lifecycle & Terminology | "Walk me through the end-to-end lifecycle of an AI model." |
| Lab 2 | AI Tooling Landscape | "Explain the modern AI tooling ecosystem and where each tool fits." |

## Module 2 — Data Science Track (Modelling Focus)

### Learning Objectives:
- Be able to perform data cleaning, exploration, and transformation
- Understand and implement basic ML and Deep Learning models
- Know how to evaluate model performance correctly
- Learn to optimise models for size, speed, and deployment constraints

### Labs:
| Lab   | Focus                               | Starter Prompt |
|-------|-------------------------------------|----------------|
| Lab 1 | Data Exploration                    | "Teach me how to clean and explore data using Pandas." |
| Lab 2 | Classical Machine Learning          | "Guide me through training a classifier with Scikit-Learn." |
| Lab 3 | Deep Learning                       | "Help me train a text classification model with PyTorch." |
| Lab 4 | Model Optimisation                  | "Show me optimisation techniques like LoRA and Quantisation." |
| Lab 5 | Evaluation & Metrics                | "Teach me how to evaluate models using proper metrics." |
| Lab 6	| Model Debugging & Inspection        | "Teach me how to trace accuracy divergence and inspect ONNX weights and classifier heads." |
| Lab 7 | Improving Quality Across All Labels | "How can I improve model quality across both custom and default labels?" |
| Lab 8 | More Complex Real-World Usage       | "Let's fine-tune an existing LLM (via LoRA) to generate domain-specific quizzes and questions aligned to a curriculum" |

## Module 3 — AI Engineering Track (Platform Focus)

### Learning Objectives:
- Be able to package and export trained models for serving
- Build production-grade inference APIs
- Deploy AI systems into Kubernetes using infrastructure as code
- Implement model versioning and lifecycle management
- Monitor models in production for health and drift

### Labs:
| Lab | Focus | Starter Prompt |
|-----|-------|----------------|
| Lab 1 | Model Packaging | "Teach me how to export models for serving using ONNX/TorchScript." |
| Lab 2 | Inference API | "Guide me to build an inference API with FastAPI and Triton." |
| Lab 3 | Kubernetes Deployment | "Show me how to deploy my inference API into Kubernetes using Helm." |
| Lab 4 | Model Versioning | "Teach me to track and version models with MLFlow or W&B." |
| Lab 5 | Monitoring & Drift Detection | "Help me monitor models with Prometheus, Grafana, and drift detection strategies." |

## Module 4 — AI Platform Architecture (Systems Thinking)

### Learning Objectives:
- Understand advanced patterns for production-grade AI systems
- Evaluate trade-offs in platform architecture (multi-tenancy, cost, governance)
- Design for scalability, reliability, and security
- Learn about RAG, Vector DBs, and large-scale search augmentation

### Labs:
| Lab | Focus | Starter Prompt |
|-----|-------|----------------|
| Lab 1 | Retrieval Augmented Generation (RAG) | "Teach me about RAG systems and Vector DB architecture." |
| Lab 2 | Multi-Tenant AI Systems | "Explain multi-tenant AI design patterns and their trade-offs." |
| Lab 3 | Cost Engineering | "Help me understand AI cost optimisation strategies at platform scale." |
| Lab 4 | Security & Governance | "Teach me security best practices for AI APIs and models." |
| Lab 5 | Model Ops Pipelines | "Compare GitOps vs MLOps for managing AI model pipelines." |

## Optional — Lab Notes & Experiments

Learning Objectives:
- Capture learnings, mistakes, experiments, design notes
- Document problems and how you solved them
- Record performance tuning, costs, infrastructure lessons
