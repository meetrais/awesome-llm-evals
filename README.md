# Awesome LLM Evals

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: No License](https://img.shields.io/badge/License-No_License-lightgrey.svg)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

> A comprehensive collection of evaluation benchmarks, datasets, and leaderboards for Large Language Models (LLMs)

As LLMs continue to evolve at a rapid pace, evaluation benchmarks have become essential tools for measuring progress, identifying limitations, and guiding development. This repository provides a detailed guide to the most important, latest, and interesting LLM evaluation benchmarks available today.

**Last Updated: November 26, 2025**

---

## Table of Contents

- [General Knowledge and Reasoning](#general-knowledge-and-reasoning)
- [Mathematical Reasoning](#mathematical-reasoning)
- [Code Generation and Software Engineering](#code-generation-and-software-engineering)
- [Scientific Knowledge](#scientific-knowledge)
- [Instruction Following](#instruction-following)
- [Conversational and Human Preference](#conversational-and-human-preference)
- [Long-Term Coherence and Agentic](#long-term-coherence-and-agentic)
- [Emotional Intelligence](#emotional-intelligence)
- [Factuality and Truthfulness](#factuality-and-truthfulness)
- [Commonsense Reasoning](#commonsense-reasoning)
- [Frontier and Expert-Level](#frontier-and-expert-level)
- [Leaderboards and Resources](#leaderboards-and-resources)
- [Contributing](#contributing)

---

## General Knowledge and Reasoning

### MMLU-Pro

| Attribute | Details |
|-----------|---------|
| **Paper** | [MMLU-Pro: A More Robust and Challenging Benchmark](https://arxiv.org/abs/2406.01574) |
| **Dataset** | [HuggingFace](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) |
| **Questions** | 12,000+ questions |
| **Answer Choices** | 10 (vs. 4 in original MMLU) |
| **Status** | Active, more challenging |

An enhanced version of MMLU designed to be more challenging and reduce the impact of random guessing by expanding answer choices from 4 to 10 and increasing question complexity.

**Key Improvements:**
- More complex reasoning required
- Reduced effectiveness of random guessing
- Better discrimination between frontier models
- Focus on reasoning over pure knowledge retrieval

---

### ARC (AI2 Reasoning Challenge)

| Attribute | Details |
|-----------|---------|
| **Paper** | [Think You Have Solved Question Answering?](https://arxiv.org/abs/1803.05457) |
| **Dataset** | [AI2 ARC](https://allenai.org/data/arc) |
| **Questions** | 7,787 grade-school science questions |
| **Subsets** | ARC-Easy (5,197) and ARC-Challenge (2,590) |
| **Format** | Multiple-choice (3-5 options) |

ARC tests models on grade-school science questions requiring reasoning beyond simple fact retrieval. The Challenge subset specifically contains questions that both word co-occurrence and retrieval algorithms answer incorrectly.

---

## Mathematical Reasoning

### AIME 2024/2025 (American Invitational Mathematics Examination)

| Attribute | Details |
|-----------|---------|
| **Paper** | Various technical reports |
| **Dataset** | [HuggingFace](https://huggingface.co/datasets/opencompass/AIME2025) |
| **Questions** | 30 problems per year (2 contests x 15) |
| **Difficulty** | Competition-level (top 5% high school students) |
| **Answer Format** | Integer from 000-999 |
| **Human Median** | 4-6 correct out of 15 |

AIME problems are derived from the prestigious American Invitational Mathematics Examination, challenging models with multi-step reasoning across algebra, geometry, number theory, and combinatorics.

---

### FrontierMath

| Attribute | Details |
|-----------|---------|
| **Paper** | [FrontierMath: A Benchmark for Evaluating Advanced Mathematical Reasoning in AI](https://arxiv.org/abs/2411.04872) |
| **Created By** | Epoch AI (with OpenAI support) |
| **Problems** | Several hundred original problems |
| **Difficulty Tiers** | 4 (Tier 4 = research-level mathematics) |
| **Contributors** | 60+ mathematicians from universities worldwide |
| **Status** | Active, highly challenging (over 98% unsolved) |

FrontierMath is designed to evaluate whether AI systems possess research-level mathematical reasoning capabilities. Problems typically require hours or even days for specialist mathematicians to solve.

---

### OlymMATH

| Attribute | Details |
|-----------|---------|
| **Paper** | [Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark](https://arxiv.org/abs/2503.21380) |
| **Problems** | 200 curated problems |
| **Subsets** | OlymMATH-EASY (AIME-level) and OlymMATH-HARD |
| **Languages** | English and Chinese versions |

A newer Olympiad-level benchmark with significantly harder problems than AIME.

---

## Code Generation and Software Engineering

### MLE-bench (Machine Learning Engineering)

| Attribute | Details |
|-----------|---------|
| **Paper** | [MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering](https://openai.com/index/mle-bench/) |
| **Created By** | OpenAI |
| **Tasks** | 75 Kaggle competitions |
| **Metric** | Medal rate (Gold, Silver, Bronze) |

MLE-bench evaluates how well AI agents perform on end-to-end machine learning engineering tasks sourced from real Kaggle competitions. It tests skills in dataset preparation, model training, and experiment execution.

**Key Features:**
- Measures "AI building AI" capabilities
- Uses real-world competitions (House Prices, Titanic, ImageNet variants, etc.)
- **Human Baseline:** Kaggle Grandmasters
- **Top Agents:** Achieve "Bronze" level in ~16.9% of competitions (o1-preview baseline).

---

### SWE-bench (Software Engineering Benchmark)

| Attribute | Details |
|-----------|---------|
| **Paper** | [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770) |
| **Dataset** | [GitHub](https://github.com/princeton-nlp/SWE-bench) |
| **Problems** | 2,294 real GitHub issues |
| **Repositories** | 12 popular Python repos |
| **Task** | Generate patches to resolve issues |
| **Subsets** | Verified (500), Lite (300), Multimodal (517) |

SWE-bench evaluates LLMs on real-world software engineering tasks. It has become the industry standard for evaluating agentic coding capabilities.

---

### LiveCodeBench

| Attribute | Details |
|-----------|---------|
| **Paper** | [LiveCodeBench: Holistic and Contamination Free Evaluation](https://arxiv.org/abs/2403.07974) |
| **Website** | [livecodebench.github.io](https://livecodebench.github.io/) |
| **Problems** | 1,150+ (release v7, Nov 2025) |
| **Sources** | LeetCode, AtCoder, CodeForces |
| **Status** | Continuously updated |

A contamination-free benchmark that continuously collects new competitive programming problems released *after* a model's training cutoff.

---

### BigCodeBench

| Attribute | Details |
|-----------|---------|
| **Paper** | [BigCodeBench: Benchmarking Code Generation](https://arxiv.org/abs/2406.15877) |
| **Dataset** | [GitHub](https://github.com/bigcode-project/bigcodebench) |
| **Tasks** | 1,140 software engineering tasks |
| **Focus** | Complex instruction following & library usage |

BigCodeBench challenges LLMs to solve practical programming tasks requiring diverse function calls (Pandas, Matplotlib, Requests) rather than just algorithmic logic.

---

## Scientific Knowledge

### GPQA Diamond (Graduate-Level Google-Proof Q&A)

| Attribute | Details |
|-----------|---------|
| **Paper** | [GPQA: A Graduate-Level Google-Proof Q&A Benchmark](https://arxiv.org/abs/2311.12022) |
| **Dataset** | [HuggingFace](https://huggingface.co/datasets/Idavidrein/gpqa) |
| **Questions** | 198 (Diamond), 448 (Main) |
| **Subjects** | Biology, Physics, Chemistry |
| **Authors** | PhD-level domain experts |

GPQA features questions written by domain experts that are "Google-proof"--designed to be extremely difficult even with unrestricted internet access.

**Status:** Approaching saturation. Analysis by Epoch AI suggests approximately 90-95% of questions are valid.

---

## Instruction Following

### IFEval (Instruction Following Evaluation)

| Attribute | Details |
|-----------|---------|
| **Paper** | [Instruction-Following Evaluation for Large Language Models](https://arxiv.org/abs/2311.07911) |
| **Focus** | Verifiable instruction following |
| **Format** | Programmatically checkable instructions |

IFEval focuses on "verifiable instructions" (e.g., "no capitalized words," "response must be JSON").

**Metrics:**
- **Strict Accuracy**: Binary--did the LLM follow instructions exactly?
- **Loose Accuracy**: Accounts for minor variations

---

## Conversational and Human Preference

### Chatbot Arena / LMArena

| Attribute | Details |
|-----------|---------|
| **Website** | [lmarena.ai](https://lmarena.ai/) |
| **Method** | Crowdsourced human evaluation |
| **Format** | Anonymous head-to-head battles |
| **Metric** | Elo rating system |
| **Votes Collected** | 5M+ |

The gold standard for human preference evaluation.

---

### WildBench

| Attribute | Details |
|-----------|---------|
| **Paper** | [WildBench: Benchmarking LLMs with Challenging Tasks](https://arxiv.org/abs/2406.04770) |
| **Website** | [huggingface.co/spaces/allenai/WildBench](https://huggingface.co/spaces/allenai/WildBench) |
| **Questions** | 1,024 challenging real-world user queries |
| **Judge** | GPT-4o / Claude 3.5 Sonnet (LLM-as-a-judge) |

WildBench evaluates models on "wild" real-world user prompts that are significantly harder and more diverse than standard Chatbot Arena prompts. It focuses on task complexity and mitigating length bias.

---

## Long-Term Coherence and Agentic

### TAU-bench

| Attribute | Details |
|-----------|---------|
| **Paper** | [TAU-bench: A Benchmark for Tool-Augmented User Simulation](https://arxiv.org/abs/2406.12045) |
| **Dataset** | [GitHub](https://github.com/sierra-research/tau-bench) |
| **Domains** | Retail and Airline |
| **Metric** | Pass Rate (max 30 turns) |

TAU-bench evaluates agents in realistic, tool-heavy scenarios (e.g., modifying a flight, returning a purchase) where the user is also simulated. It tests the ability to navigate complex policies and database states reliably.

**Performance:**
- Frontier models often struggle to exceed 60-70% success rates due to strict policy adherence requirements.

---

### Vending-Bench 2

| Attribute | Details |
|-----------|---------|
| **Paper** | [Vending-Bench: A Benchmark for Long-Term Coherence](https://arxiv.org/abs/2502.15840) |
| **Website** | [andonlabs.com/evals/vending-bench](https://andonlabs.com/evals/vending-bench-2) |
| **Task** | Manage a vending machine business for a simulated year |
| **Status** | Active (V2) |

Vending-Bench 2 is a benchmark for measuring AI model performance on running a business over long time horizons. Models are tasked with running a simulated vending machine business over a year and scored on their bank account balance at the end.
---

### GAIA (General AI Assistants)

| Attribute | Details |
|-----------|---------|
| **Paper** | [GAIA: a benchmark for General AI Assistants](https://arxiv.org/abs/2311.12983) |
| **Dataset** | [HuggingFace](https://huggingface.co/gaia-benchmark) |
| **Human Performance** | 92% |

GAIA tests AI assistants on real-world questions requiring reasoning, multi-modality handling, web browsing, and tool-use.

---

## Emotional Intelligence

### EQ-Bench 3

| Attribute | Details |
|-----------|---------|
| **Paper** | [EQ-Bench: An Emotional Intelligence Benchmark](https://arxiv.org/abs/2312.06281) |
| **Website** | [eqbench.com](https://eqbench.com/) |
| **Questions** | 45 challenging roleplay scenarios |
| **Judge** | Claude Sonnet 3.7 |

EQ-Bench 3 evaluates emotional intelligence in LLMs by testing their ability to understand complex emotions and social interactions.

---

## Factuality and Truthfulness

### SimpleQA / SimpleQA Verified

| Attribute | Details |
|-----------|---------|
| **Paper** | [SimpleQA Verified](https://arxiv.org/abs/2509.07968) |
| **Created By** | OpenAI |
| **Focus** | Short-form factual accuracy (hallucination rate) |

SimpleQA measures LLMs' parametric factuality--their ability to correctly answer short factual questions from their training knowledge (not retrieval).

---

## Commonsense Reasoning

### HellaSwag & WinoGrande

| Benchmark | Details |
|-----------|---------|
| **HellaSwag** | Tests commonsense natural language inference via sentence completion (95%+ accuracy now common). |
| **WinoGrande** | Evaluates pronoun resolution in adversarial contexts. |

---

## Frontier and Expert-Level

### Humanity's Last Exam (HLE)

| Attribute | Details |
|-----------|---------|
| **Paper** | [Humanity's Last Exam](https://arxiv.org/abs/2501.14249) |
| **Website** | [agi.safe.ai](https://agi.safe.ai/) |
| **Questions** | 2,500 finalized (public) + private test set |
| **Contributors** | Nearly 1,000 experts from 500+ institutions |
| **Status** | Active, extremely hard |

Designed to be "the final closed-ended academic benchmark," HLE contains extremely challenging questions at the frontier of human knowledge across 100+ subjects.

---

## Leaderboards and Resources

### Major Leaderboards

| Leaderboard | Focus | URL |
|-------------|-------|-----|
| **LMArena (Chatbot Arena)** | Human preference (5M+ votes) | [lmarena.ai](https://lmarena.ai/) |
| **Vellum LLM Leaderboard** | Multi-benchmark (excludes saturated benchmarks) | [vellum.ai/llm-leaderboard](https://www.vellum.ai/llm-leaderboard) |
| **Open LLM Leaderboard** | Open-source models | [vellum.ai/open-llm-leaderboard](https://www.vellum.ai/open-llm-leaderboard) |
| **SEAL Leaderboards** | Expert-driven (HLE, SWE-bench Pro) | [scale.com/leaderboard](https://scale.com/leaderboard) |