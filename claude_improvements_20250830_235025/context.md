# Repository Context

## Structure:
./.terragon/backlog.md
./.terragon/scoring-model.py
./ARCHITECTURE.md
./auth_config.py
./autonomous_quality_gates.py
./autonomous_quality_gates_comprehensive.py
./autonomous_quality_gates_final.py
./AUTONOMOUS_SDLC_COMPLETE_IMPLEMENTATION.md
./AUTONOMOUS_SDLC_COMPLETION_REPORT.md
./AUTONOMOUS_SDLC_EXECUTION_SUMMARY.md
./AUTONOMOUS_SDLC_IMPLEMENTATION_COMPLETE.md
./AUTONOMOUS_SDLC_PRODUCTION_DEPLOYMENT.md
./AUTONOMOUS_SDLC_REPORT.md
./AUTONOMOUS_VALUE_DISCOVERY.md
./build/scripts/generate-sbom.py
./CHANGELOG.md
./CHECKPOINTED_SDLC_IMPLEMENTATION_SUMMARY.md
./CODE_OF_CONDUCT.md
./comprehensive_quality_gates.py
./comprehensive_quality_gates_v2.py

## README (if exists):
# quantum-inspired-task-planner

[![Build Status](https://img.shields.io/github/actions/workflow/status/danieleschmidt/quantum-inspired-task-planner/ci.yml?branch=main)](https://github.com/danieleschmidt/quantum-inspired-task-planner/actions)
[![Security Score](https://img.shields.io/badge/security_score-8.7%2F10-green.svg)](docs/SETUP_COMPLETE.md)
[![SDLC Maturity](https://img.shields.io/badge/SDLC_maturity-95%2F100-brightgreen.svg)](docs/SETUP_COMPLETE.md)
[![Test Coverage](https://img.shields.io/badge/coverage-92%25-green.svg)](tests/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Quantum](https://img.shields.io/badge/quantum-Azure%20|%20D--Wave%20|%20IBM-purple)](https://github.com/danieleschmidt/quantum-inspired-task-planner)

**Enterprise-grade QUBO-based task scheduler for agent pools.** Solves complex assignment problems using quantum annealing, gate-based quantum computing, or classical simulators with comprehensive SDLC automation, security scanning, and operational excellence.

> 🚀 **Production-Ready**: This repository features a complete enterprise-grade SDLC implementation with automated testing, security scanning, performance monitoring, and operational procedures. [See implementation details →](docs/SETUP_COMPLETE.md)

## 🎯 Key Features

- **QUBO Formulation**: Automatic conversion of constraints to quantum format
- **Multi-Backend**: Azure Quantum, D-Wave, IBM Quantum, and simulators
- **Agent Framework Integration**: Native support for CrewAI, AutoGen, LangChain
- **Hybrid Classical-Quantum**: Seamless fallback for problems of any size
- **Real-time Adaptation**: Dynamic re-scheduling based on task completion
- **Cost Optimization**: Automatic backend selection based on problem complexity

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [SDLC & Operations](#sdlc--operations)
- [Problem Formulation](#problem-formulation)
- [Quantum Backends](#quantum-backends)
- [Agent Integration](#agent-integration)
- [Classical Fallbacks](#classical-fallbacks)
- [Optimization Strategies](#optimization-strategies)
- [Benchmarks](#benchmarks)
- [API Reference](#api-reference)
- [Contributing](#contributing)

## 🚀 Installation

### From PyPI

```bash
pip install quantum-inspired-task-planner
```

### With Quantum Backends

```bash
# Install with specific quantum providers
pip install quantum-inspired-task-planner[dwave]

## Main files:
