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
| **Questions** | 12,000+ rigorously curated questions |
| **Sources** | Academic exams and textbooks |
| **Domains** | 14 (Biology, Business, Chemistry, CS, Economics, Engineering, Health, History, Law, Math, Philosophy, Physics, Psychology, Others) |
| **Answer Choices** | 10 (vs. 4 in original MMLU) |
| **Status** | Active, more challenging |

MMLU-Pro is an enhanced benchmark designed to evaluate language understanding models across broader and more challenging tasks. Building on the Massive Multitask Language Understanding (MMLU) dataset, MMLU-Pro integrates more challenging, reasoning-focused questions and significantly raises the difficulty by expanding answer choices from four to ten options, substantially reducing the chance of success through random guessing.

**What's New About MMLU-Pro:**

**1. Increased Answer Options (4 → 10)**
- Makes evaluation more realistic and challenging
- Dramatically reduces random guessing success rate
- Improves benchmark robustness

**2. Reasoning-Focused Questions**
- Original MMLU contains mostly knowledge-driven questions without requiring much reasoning
- MMLU-Pro increases problem difficulty and integrates more reasoning-focused problems
- **Chain-of-Thought (CoT) prompting can be 20% higher than PPL** (Perplexity-based) methods, demonstrating the reasoning emphasis
- In original MMLU, PPL results are normally better than CoT

**3. Enhanced Robustness to Prompt Variations**
- With 24 different prompt styles tested, sensitivity to prompt variations decreased from **4-5% in MMLU to just 2% in MMLU-Pro**
- Provides more stable and reliable evaluation across different prompting strategies

---

### ARC (AI2 Reasoning Challenge & ARC-AGI-2)

**ARC (AI2 Reasoning Challenge)**

| Attribute | Details |
|-----------|---------|
| **Paper** | [Think You Have Solved Question Answering?](https://arxiv.org/abs/1803.05457) |
| **Dataset** | [AI2 ARC](https://allenai.org/data/arc) |
| **Questions** | 7,787 grade-school science questions |
| **Subsets** | ARC-Easy (5,197) and ARC-Challenge (2,590) |
| **Format** | Multiple-choice (3-5 options) |

ARC (the AI2 Reasoning Challenge) is a multiple-choice science QA benchmark of 7,787 natural, grade-school questions, partitioned into an Easy set and a more difficult Challenge set that standard retrieval and word co-occurrence baselines fail to answer.

**Structure and subsets**  
Subsets: ARC-Easy (5,197 questions) and ARC-Challenge (2,590 questions), where the Challenge set contains only items missed by both a retrieval-based system and a word co-occurrence/PMI baseline.​

**Format**  
Each question has 3–5 answer options and is scored by accuracy; an associated 14M+ sentence science corpus is provided to support knowledge-intensive models.​

**What ARC measures?**  
ARC is designed to probe reasoning that goes beyond surface pattern matching, including multi-hop inference, integration of background science knowledge, and non-trivial distractor handling.​

The Challenge subset is widely used as a benchmark for advanced QA systems because it filters out questions solvable by simple lexical overlap or retrieval heuristics, highlighting true reasoning and knowledge integration capabilities.​

---

**ARC-AGI-2 (Abstraction and Reasoning Corpus for AGI)**

| Attribute | Details |
|-----------|---------|
| **Technical Paper** | [ARC-AGI-2 Technical Paper](https://arcprize.org/arc-agi-2) |
| **Introduced** | 2025 (ARC-AGI-1 launched in 2019) |
| **Tasks** | 1,400+ unique tasks across multiple evaluation sets |
| **Human Testing** | 400 participants tested |
| **Human Success** | At least 2 humans solved every task in ≤2 attempts |
| **Average Time** | ~2.3 minutes per task (human) |
| **Status** | Active, extremely challenging for AI |

ARC-AGI-2 is the second edition benchmark designed to measure general fluid intelligence and abstract reasoning capabilities in AI systems. Unlike traditional benchmarks, ARC-AGI-2 is specifically crafted to be "easy for humans, hard for AI"—every task is solvable by humans with minimal prior knowledge, yet remains out of reach for current frontier AI systems.

**Evolution from ARC-AGI-1:**

ARC-AGI-1 (2019) was designed to challenge deep learning by resisting simple memorization. The training dataset teaches Core Knowledge Priors required to solve novel evaluation tasks. Think of it as learning basic math symbols (training) and then solving algebra equations (evaluation)—you cannot memorize your way to answers, you must apply existing knowledge to new problems.

ARC-AGI-2 significantly raises the bar by demanding both **high adaptability and high efficiency**, providing more granular evaluation signals for next-generation AI systems.

**Design Goals for ARC-AGI-2:**

1. **Same Fundamental Principles** - Each task is unique and cannot be memorized, requiring only elementary Core Knowledge
2. **Less Brute-Forcible** - Designed to minimize susceptibility to naive or computationally intensive brute-force program search
3. **First-Party Human Testing** - Direct empirical comparison between human and AI performance
4. **More "Signal Bandwidth"** - Wider range of scores to measure the capability gap toward AGI
5. **Calibrated Difficulty** - Consistent difficulty distributions across Public, Private, and Semi-Private evaluation sets

**Three Key Conceptual Challenges:**

**1. Symbolic Interpretation**
- Frontier AI reasoning systems struggle with tasks requiring symbols to have meaning beyond visual patterns
- Systems attempt pattern matching (symmetry, mirroring, transformations) but fail to assign semantic significance to symbols

**2. Compositional Reasoning**
- AI systems struggle with simultaneous application of multiple interacting rules
- Can handle single or few global rules consistently, but fail when rules must work together

**3. Contextual Rule Application**
- Systems struggle when rules must be applied differently based on context
- Tendency to fixate on superficial patterns rather than understanding underlying selection principles

**Human Performance Validation:**

- **400 participants** tested on **1,400+ unique tasks**
- Every task met the threshold: **at least 2 humans solved it in ≤2 attempts**
- Average ~9-10 participants attempted each task
- No significant correlation found between demographic factors and performance, suggesting ARC-AGI-2 assesses general problem-solving rather than domain-specific knowledge

---

## Mathematical Reasoning

### AIME 2024/2025 (American Invitational Mathematics Examination)

| Attribute | Details |
|-----------|------------|
| **Paper** | Various technical reports |
| **Dataset** | [HuggingFace](https://huggingface.co/datasets/opencompass/AIME2025) |
| **Questions** | 30 problems per year (2 contests x 15) |
| **Competition Type** | Invite-only, gateway to USAMO |
| **Difficulty** | Competition-level (top 5% high school students) |
| **Answer Format** | Single integer from 000-999 |
| **Human Median** | 4-6 correct out of 15 |

The American Invitational Mathematics Examination (AIME) is a prestigious, invite-only mathematics competition for high school students that serves as a crucial gateway for qualifying to the USA Mathematical Olympiad (USAMO). This benchmark evaluates LLM performance on highly challenging problems covering a wide range of mathematical topics including algebra, geometry, number theory, and combinatorics.

**Key Characteristics:**

**Difficulty and Challenge Level**
- Highly challenging problems requiring multi-step reasoning and deep mathematical understanding
- Each answer is a single integer (000-999), requiring exact precision
- Median human score: **4-6 correct answers out of 15**, demonstrating the exceptional difficulty
- Problems demand sophisticated problem-solving strategies beyond pattern recognition

**Challenging Benchmark**
- AIME remains a challenging benchmark for evaluating mathematical reasoning in AI systems
- While some frontier models achieve strong performance, the benchmark continues to provide headroom for improvement
- Serves as a useful discriminator between models' mathematical reasoning capabilities
- The difficulty level ensures meaningful evaluation of advanced mathematical problem-solving abilities

**Data Contamination Concerns**
- All AIME questions and answers are publicly available, creating potential for data contamination if included in model pretraining corpora
- Models perform **significantly better on older AIME versions compared to newer ones**, raising suspicions about memorization vs. genuine reasoning
- This performance gap highlights the importance of using recent AIME editions for fair evaluation
- Evaluators should be cautious when interpreting results and consider testing on the most recent years

> **Note:** When using AIME for evaluation, prioritize the most recent exam years (2024/2025) to minimize potential data contamination effects and obtain more accurate assessments of true mathematical reasoning capabilities.

---

### FrontierMath

| Attribute | Details |
|-----------|---------|
| **Paper** | [FrontierMath: A Benchmark for Evaluating Advanced Mathematical Reasoning in AI](https://arxiv.org/abs/2411.04872) |
| **Created By** | Epoch AI (with OpenAI support) |
| **Problems** | 350 original mathematics problems |
| **Difficulty Range** | Challenging university-level to expert research problems |
| **Difficulty Tiers** | 4 (Tier 4: 50 extremely difficult problems) |
| **Contributors** | Mathematics professors and postdoctoral researchers |
| **Format** | Natural-language problems with closed-form expressions |
| **Status** | Novel, unpublished problems |

FrontierMath evaluates whether AI systems possess research-level mathematical reasoning capabilities. The benchmark spans from challenging university-level questions to problems that may take expert mathematicians days to solve, demanding creative insight, connecting disparate concepts, and sophisticated reasoning.

**Key Characteristics:**

**Problem Design and Requirements**
- Natural-language math problems with solutions expressed as closed-form mathematical expressions
- All problems are novel and unpublished to ensure models haven't encountered them during training
- **Definite, verifiable answers**: Large integers, symbolic reals, or tuples that can be checked computationally
- **"Guessproof" design**: Answers resist random attempts or trivial brute-force approaches
- **Computational tractability**: Solution scripts must run in less than one minute on standard hardware

**Tier 4 - Most Challenging**
- 50 extremely difficult problems developed as short-term research projects
- Created by mathematics professors and postdoctoral researchers
- Solving these tasks would provide evidence that AI can perform the complex reasoning needed for scientific breakthroughs

**Quality Assurance**
- Each problem undergoes peer review by expert mathematicians
- Reviews verify correctness, check for ambiguities, and assess difficulty ratings
- Error rate approximately 1 in 20 problems (comparable to ImageNet and other major ML benchmarks)
- Expanding expert review process and error-bounty program to further reduce errors

---

### OlymMATH

| Attribute | Details |
|-----------|---------|
| **Paper** | [Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark](https://arxiv.org/abs/2503.21380) |
| **Dataset** | [GitHub](https://github.com/RUCAIBox/OlymMATH) |
| **Problems** | 200 meticulously curated problems |
| **Subsets** | AIME-level (easy) and Harder problems |
| **Languages** | English (OlymMATH-EN) and Chinese (OlymMATH-ZH) |
| **Sources** | Manually sourced from printed materials (magazines, textbooks, competition materials) |
| **Focus** | Four core mathematical fields |

OlymMATH is a new, challenging Olympiad-level mathematical benchmark for evaluating the complex reasoning capabilities of large language models (LLMs). It was introduced by researchers to address the saturation of existing, easier math benchmarks and minimize data contamination risks.

**Key Features of OlymMATH:**

**Content and Difficulty**
- The benchmark comprises 200 meticulously curated problems sourced manually from printed materials (magazines, textbooks, competition materials) to ensure originality and minimize prior online exposure
- The problems are split into two tiers:
  - **AIME-level (easy)**: Establishes a baseline for current LLMs
  - **Harder problems**: Designed to push the boundaries of state-of-the-art models, with which even top models struggle

**Bilingual Assessment**
- All problems are available in parallel English (OlymMATH-EN) and Chinese (OlymMATH-ZH) versions, enabling cross-lingual evaluation of reasoning abilities

**Objective Evaluation**
- Each problem includes a verifiable numerical solution, allowing for objective, rule-based, and automatic scoring

**Focus Areas**
- The problems span four core mathematical fields
    - Algebra
    - Geometry
    - Number theory
    - Combinatorics

---

## Code Generation and Software Engineering

### MLE-bench (Machine Learning Engineering)

| Attribute | Details |
|-----------|---------|
| **Paper** | [MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering](https://openai.com/index/mle-bench/) |
| **Created By** | OpenAI |
| **Tasks** | 75 ML engineering-related Kaggle competitions |
| **Metric** | Medal rate (Gold, Silver, Bronze) |
| **Code** | [Open-source benchmark code](https://github.com/openai/mle-bench) |

OpenAI introduced MLE-bench, a benchmark for measuring how well AI agents perform at machine learning engineering. The benchmark curates 75 ML engineering-related competitions from Kaggle, creating a diverse set of challenging tasks that test real-world ML engineering skills such as training models, preparing datasets, and running experiments.

**Key Features:**

**Comprehensive Real-World Tasks**
- 75 curated ML engineering competitions from Kaggle
- Tests diverse ML engineering skills: dataset preparation, model training, experiment execution
- Measures "AI building AI" capabilities through end-to-end ML workflows

**Human Baseline Establishment**
- Human baselines established for each competition using Kaggle's publicly available leaderboards
- Enables direct comparison between AI agent performance and human ML practitioners
- Provides meaningful performance benchmarks across different medal tiers (Bronze, Silver, Gold)

**Evaluation and Results**
- Evaluated several frontier language models using open-source agent scaffolds
- **Best Performance:** OpenAI's o1-preview with AIDE scaffolding achieves at least the level of a Kaggle bronze medal in **16.9% of competitions**
- Demonstrates current capabilities and limitations of AI agents in ML engineering tasks

**Additional Research Investigations**
- **Resource Scaling:** Investigates various forms of resource-scaling for AI agents to understand how additional compute or data affects performance
- **Contamination Analysis:** Examines the impact of contamination from pre-training data on benchmark results

**Open-Source Availability**
- Benchmark code is open-sourced to facilitate future research
- Enables the research community to understand and improve the ML engineering capabilities of AI agents

---

### SWE-bench (Software Engineering Benchmark)

| Attribute | Details |
|-----------|---------|
| **Paper** | [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770) |
| **Website** | [swebench.com](https://www.swebench.com/original.html) |
| **Dataset** | [GitHub](https://github.com/princeton-nlp/SWE-bench) |
| **Problems** | 2,294 real GitHub issues |
| **Repositories** | 12 popular Python repos |
| **Task** | Generate patches to resolve issues |
| **Subsets** | Verified (500), Lite (300), Multimodal (517) |
| **Released** | October 2023 |

SWE-bench tests AI systems' ability to solve real-world GitHub issues. It has become the industry standard for evaluating agentic coding capabilities.

**Task Collection Methodology:**

The benchmark collects 2,294 task instances by crawling Pull Requests and Issues from 12 popular Python repositories. Each instance is based on a pull request that:
1. Is associated with an issue
2. Modified 1+ testing related files

**Execution Environment:**

Per instance, an execution environment (Docker Image) is constructed with the repository successfully installed at the commit that the Pull Request is based on:
- **Without the Pull Request's changes:** A number of test(s) fail
- **After the Pull Request is merged:** The same set of test(s) pass
- These "Fail-to-Pass" tests are the primary signal for evaluation

**Evaluation Process:**

SWE-bench evaluation works as follows:
1. Per task instance, an AI system is given the issue text
2. The AI system should then modify the codebase to resolve the described issues
3. When the AI system finishes, the aforementioned Fail-to-Pass tests are run to check if the issue was successfully resolved

**Performance and Evolution:**

- **Initial Baseline (October 2023):** Retrieval Augmented Generation (RAG) baseline scored just **1.96%**
- **SWE-agent:** The first agent-based AI system introduced for performing software engineering tasks, achieving **12.47%** on SWE-bench
- **SWE-smith Dataset:** Available for training agentic software engineering models

---

### LiveCodeBench

| Attribute | Details |
|-----------|---------|
| **Paper** | [LiveCodeBench: Holistic and Contamination Free Evaluation](https://arxiv.org/abs/2403.07974) |
| **Website** | [livecodebench.github.io](https://livecodebench.github.io/) |
| **Problems** | 1,150+ (release v7, Nov 2025) |
| **Initial Collection** | 300+ problems (May 2023 - February 2024) |
| **Sources** | LeetCode, AtCoder, CodeForces |
| **Models Evaluated** | 29 LLMs |
| **Status** | Continuously updated |

LiveCodeBench is a holistic and contamination-free evaluation benchmark for LLMs that continuously collects new problems over time. Unlike traditional benchmarks, LiveCodeBench focuses on broader code-related capabilities beyond mere code generation, including self-repair, code execution, and test output prediction.

**Key Features:**

**Holistic Evaluation**
- Evaluates models on a variety of code-related scenarios:
  - **Code Generation:** Traditional code writing tasks
  - **Self-Repair:** Ability to fix broken code
  - **Test Output Prediction:** Predicting test results without execution
  - **Code Execution:** Understanding runtime behavior
- While model performances are correlated across different scenarios, relative performances and ordering can vary between scenarios

**Contamination-Free Methodology**

LiveCodeBench addresses the critical issue of data contamination through time-based evaluation:
- **Release Date Annotation:** All problems are annotated with their original release dates
- **Post-Cutoff Evaluation:** For a model with training-cutoff date D, it can be evaluated on problems released after D to measure generalization on truly unseen problems
- **Continuous Updates:** Regularly adds new problems to maintain freshness and prevent contamination

**Empirical Findings**

The benchmark reveals important insights about model performance and potential contamination:

- **DeepSeek Performance Pattern:** DeepSeek models exhibit a stark drop in performance on LeetCode problems released since September 2023 (its release date), indicating earlier problems might be contaminated
- **GPT Stability:** GPT models show relatively stable performance across different months, suggesting better generalization
- **Model Comparison:** Testing 29 LLMs across different time periods reveals novel empirical findings not captured in traditional static benchmarks

**Time-Based Analysis**
- Performance tracking across different release months enables detection of contamination patterns
- Allows measuring true generalization capabilities vs. memorization
- Provides more reliable evaluation for comparing models with different training cutoff dates

---

### BigCodeBench

| Attribute | Details |
|-----------|---------|
| **Paper** | [BigCodeBench: Benchmarking Code Generation](https://arxiv.org/abs/2406.15877) |
| **Website** | [Leaderboard](https://bigcode-bench.github.io/) |
| **Dataset** | [GitHub](https://github.com/bigcode-project/bigcodebench) |
| **Tasks** | 1,140 software engineering tasks |
| **Splits** | Complete (full docstrings) & Instruct (natural language) |
| **Focus** | Complex instruction following & library usage |

BigCodeBench is an easy-to-use benchmark for solving practical and challenging tasks via code. It aims to evaluate the true programming capabilities of large language models (LLMs) in a more realistic setting. The benchmark is designed for HumanEval-like function-level code generation tasks, but with much more complex instructions and diverse function calls.

**Evaluation Splits:**

BigCodeBench provides two distinct evaluation approaches:

1. **Complete Split**
   - Designed for code completion based on comprehensive docstrings
   - Models generate code using detailed documentation
   - Tests ability to understand and implement from specifications

2. **Instruct Split**
   - Works specifically for instruction-tuned and chat models
   - Models generate code snippets based on natural language instructions
   - Instructions contain only necessary information
   - Requires more complex reasoning and interpretation

**Why BigCodeBench?**

BigCodeBench focuses on task automation via code generation with diverse function calls and complex instructions:

- **Precise Evaluation & Ranking:** Features a comprehensive [leaderboard](https://bigcode-bench.github.io/) showing latest LLM rankings before and after rigorous evaluation
- **Pre-generated Samples:** Accelerates code intelligence research by open-sourcing LLM-generated samples for various models, eliminating the need to re-run expensive benchmarks
- **Practical Tasks:** Challenges LLMs to solve real-world programming tasks requiring diverse function calls (Pandas, Matplotlib, Requests) rather than just algorithmic logic

**Industry Adoption:**

BigCodeBench has been trusted by many leading LLM teams including:
- Zhipu AI
- Alibaba Qwen
- DeepSeek
- Amazon AWS AI
- Snowflake AI Research
- ServiceNow Research
- Meta AI
- Cohere AI
- Sakana AI
- Allen Institute for Artificial Intelligence (AI2)

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
| **Format** | Multiple-choice (4 options) |
| **Random Baseline** | 25% |
| **Human Expert Score** | 69.7% (PhD-level) |

The GPQA (Graduate-Level Google-Proof Q&A) dataset is a collection of challenging multiple-choice questions in biology, physics, and chemistry. Questions are written by domain experts (people with or pursuing PhDs in the relevant fields), and they are designed to be very difficult for non-experts to answer, even with unrestricted internet access.

**GPQA Diamond Subset:**

Evaluations are typically run on the Diamond subset of GPQA, which represents a higher-quality, more challenging subset of the main GPQA dataset. The Diamond subset contains **198 questions** that meet stringent quality criteria:
- Both domain expert annotators got the correct answers
- The majority of non-domain experts answered incorrectly

This dual requirement ensures that questions are both technically accurate and genuinely challenging for those without specialized expertise.

**Format and Baselines:**

- **Question Format:** Multiple-choice questions with four options
- **Random Guessing Baseline:** 25% accuracy
- **Human Expert Performance:** OpenAI recruited PhD-level experts to answer questions in GPQA Diamond and found they scored **69.7%**

This human baseline was established as part of OpenAI's o1 model evaluation, providing a meaningful comparison point for AI performance against top human experts in their respective fields.

**Quality and Status:**

Analysis by Epoch AI suggests approximately **90-95% of questions are valid**, indicating high overall quality. The benchmark is approaching saturation as frontier models continue to improve.

---

## Instruction Following

### IFEval (Instruction Following Evaluation)

| Attribute | Details |
|-----------|---------|
| **Paper** | [Instruction-Following Evaluation for Large Language Models](https://arxiv.org/abs/2311.07911) |
| **Dataset** | [HuggingFace](https://huggingface.co/datasets/google/IFEval) |
| **Created By** | Google |
| **Instruction Types** | 25 types of verifiable instructions |
| **Prompts** | ~500 prompts |
| **Focus** | Verifiable instruction following |
| **Format** | Programmatically checkable instructions |

One core capability of Large Language Models (LLMs) is to follow natural language instructions. However, the evaluation of such abilities is not standardized: Human evaluations are expensive, slow, and not objectively reproducible, while LLM-based auto-evaluation is potentially biased or limited by the ability of the evaluator LLM.

**Benchmark Design:**

To overcome these evaluation challenges, IFEval (Instruction-Following Eval) provides a straightforward and easy-to-reproduce evaluation benchmark. It focuses on a set of "verifiable instructions" that can be programmatically checked, such as:
- "Write in more than 400 words"
- "Mention the keyword of AI at least 3 times"
- "No capitalized words"
- "Response must be JSON"

**Dataset Composition:**

IFEval identifies **25 types of verifiable instructions** and constructs around **500 prompts**, with each prompt containing one or more verifiable instructions. This structure allows for:
- Objective, automated evaluation without human judgment
- Reproducible results across different evaluations
- Unbiased assessment independent of evaluator model capabilities

**Evaluation Metrics:**

- **Strict Accuracy**: Binary metric—did the LLM follow instructions exactly?
- **Loose Accuracy**: Accounts for minor variations while maintaining instruction compliance

The benchmark's verifiable nature makes it an essential tool for objectively measuring how well LLMs can follow specific, measurable instructions without the subjectivity of human evaluation or the limitations of LLM-as-judge approaches.

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
