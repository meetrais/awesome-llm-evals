# Awesome LLM Evals

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: No License](https://img.shields.io/badge/License-No_License-lightgrey.svg)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

> A comprehensive collection of evaluation benchmarks, datasets, and leaderboards for Large Language Models (LLMs)

As LLMs continue to evolve at a rapid pace, evaluation benchmarks have become essential tools for measuring progress, identifying limitations, and guiding development. This repository provides a detailed guide to the most important, latest, and interesting LLM evaluation benchmarks available today.

**Last Updated: December 11, 2025**

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
| **Paper** | [Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/abs/2403.04132) |
| **Website** | [lmarena.ai](https://lmarena.ai/) |
| **Created By** | UC Berkeley researchers |
| **Method** | Crowdsourced human evaluation |
| **Format** | Anonymous head-to-head battles |
| **Metric** | Elo rating system |
| **Votes Collected** | 5M+ |

Created by researchers from UC Berkeley, LMArena is an open platform where everyone can easily access, explore and interact with the world's leading AI models. By comparing them side by side and casting votes for the better response, the community helps shape a public leaderboard, making AI progress more transparent and grounded in real-world usage.

**Chatbot Arena (Core Platform):**

Chatbot Arena is an open-source platform for evaluating large language models (LLMs) through crowdsourced human preferences:

- **Methodology:** Users anonymously vote on which of two chatbots provides a better response to a given prompt through blind, side-by-side comparisons
- **Ranking:** Votes are used to calculate an Elo rating for each model, which ranks them on the leaderboard
- **Data:** The platform has collected a large amount of data through user votes, which can be used for other research
- **Openness:** It is an open platform widely cited by LLM developers and researchers

**LMArena Benchmark:**

While "LMArena" often refers to the platform itself, it also represents a specific benchmark/leaderboard that has faced some criticism:

- **Data Asymmetries:** Some studies suggest the LMArena benchmark may favor larger, proprietary models due to data asymmetries and private testing privileges
- **Data Access Concerns:** There are concerns that leading model providers have had better access to the data
- **Model Version Discrepancies:** Models submitted to the leaderboard may not represent the exact versions released publicly

**Advantages and Benefits:**

- **Avoids Traditional Benchmark Pitfalls:** Unlike benchmarks that rely solely on automated scores, the Arena approach incorporates human preference in a conversational context
- **Real-World Usage:** Captures how models perform on actual user queries rather than curated test sets
- **Community-Driven:** Democratizes LLM evaluation through broad community participation

**Risks and Limitations:**

- **Risk of Gaming:** As with any benchmark, there is a risk of models being "over-optimized" for the platform, leading to distorted results (Goodhart's Law: "When a measure becomes a target, it ceases to be a good measure")
- **Potential Biases:** Possible systematic biases toward certain model types or providers
- **Need for Multiple Metrics:** The controversy highlights that no single evaluation metric should be trusted in isolation—a combination of approaches is necessary for comprehensive evaluation

**Status:** Widely regarded as the gold standard for human preference evaluation, while also requiring critical interpretation alongside other benchmarks.

---

### WildBench

| Attribute | Details |
|-----------|---------|
| **Paper** | [WildBench: Benchmarking LLMs with Challenging Tasks](https://arxiv.org/abs/2406.04770) |
| **Website** | [HuggingFace](https://huggingface.co/spaces/allenai/WildBench) |
| **GitHub** | [allenai/WildBench](https://github.com/allenai/WildBench) |
| **Created By** | Allen Institute for AI & University of Washington |
| **Tasks** | 1,024 challenging real-world user queries |
| **Source** | Selected from 1M+ human-chatbot conversation logs |
| **Judge** | GPT-4-turbo / Claude 3.5 Sonnet (LLM-as-a-judge) |
| **Metrics** | WB-Reward (0.98 correlation) & WB-Score (0.95 correlation) |

WildBench is an automated evaluation framework designed to benchmark large language models (LLMs) using challenging, real-world user queries. It consists of 1,024 tasks carefully selected from over one million human-chatbot conversation logs, focusing on "wild" prompts that are significantly harder and more diverse than standard benchmark tasks.

**Evaluation Metrics:**

WildBench has developed two advanced metrics computable using LLMs like GPT-4-turbo:

**1. WB-Reward (Pairwise Comparison)**
- Employs fine-grained pairwise comparisons between model responses
- Generates **five potential outcomes:**
  - Much better
  - Slightly better
  - Slightly worse
  - Much worse
  - Tie
- Uses **three baseline models** at varying performance levels (unlike previous evaluations with a single baseline)
- Achieves **Pearson correlation of 0.98** with Chatbot Arena's human-voted Elo ratings for top-ranking models

**2. WB-Score (Individual Quality Assessment)**
- Evaluates the quality of model outputs individually
- Fast and cost-efficient evaluation metric
- Achieves **Pearson correlation of 0.95** with Chatbot Arena Elo ratings
- **Outperforms competing benchmarks:**
  - ArenaHard: 0.91 correlation
  - AlpacaEval2.0: 0.89 (length-controlled win rates)
  - AlpacaEval2.0: 0.87 (regular win rates)

**Methodology and Features:**

**Task-Specific Checklists**
- Uses systematic checklists to evaluate model outputs
- Provides structured explanations that justify scores and comparisons
- Results in more reliable and interpretable automatic judgments

**Length Bias Mitigation**
- Implements a simple yet effective method to mitigate length bias
- Converts outcomes of "slightly better/worse" to "tie" if the winner response exceeds the loser by more than K characters
- Ensures fair evaluation regardless of response length

**Data Quality**
- Tasks selected from real-world conversation logs ensure practical relevance
- Focuses on hard tasks that better discriminate between model capabilities
- Strong correlation with human preferences demonstrates validity

WildBench's combination of real-world task complexity, sophisticated evaluation metrics, and strong correlation with human judgment makes it a powerful tool for evaluating LLMs on challenging, practical scenarios.

---

## Long-Term Coherence and Agentic

### TAU-bench

| Attribute | Details |
|-----------|---------|
| **Paper** | [TAU-bench: A Benchmark for Tool-Augmented User Simulation](https://arxiv.org/abs/2406.12045) |
| **Website** | [Sierra AI Research](https://sierra.ai/blog/benchmarking-ai-agents) |
| **Dataset** | [GitHub](https://github.com/sierra-research/tau-bench) |
| **Created By** | Sierra AI Research Team |
| **Domains** | τ-retail and τ-airline |
| **Metric** | Pass Rate & pass^k (reliability metric) |
| **Max Turns** | 30 turns per interaction |

Sierra's AI research team presents τ-bench (TAU-bench), a benchmark for evaluating AI agents' performance and reliability in real-world settings with dynamic user and tool interaction. Unlike existing benchmarks that only evaluate single-round interactions, τ-bench tests agents on completing complex tasks while interacting with LLM-simulated users and tools over multiple turns to gather required information.

**The Gap in Existing Benchmarks:**

While benchmarks like WebArena, SWE-bench, and AgentBench are useful for revealing high-level capabilities, they fall short in critical areas:
- Only evaluate a single round of interaction where all information is exchanged at once
- Don't reflect real-life scenarios where agents gather information over multiple, dynamic exchanges
- Focus on first-order statistics (average performance) without measures of reliability or adaptability

**Three Critical Requirements for Real-World Agents:**

Drawing from experience with live agents in production, τ-bench addresses three key requirements:

1. **Multi-Turn Interaction:** Agents must interact seamlessly with both humans and programmatic APIs over long horizons to incrementally gather information and solve complex problems
2. **Policy Adherence:** Agents must accurately follow complex domain-specific policies or rules to avoid violating company policies or producing unwanted behavior
3. **Reliability at Scale:** Agents must maintain consistency and reliability across millions of interactions to ensure predictable behavior

**Modular Framework Components:**

τ-bench uses a modular framework with three key elements:

1. **Realistic Databases and Tool APIs:** Complex, stateful databases with programmatic APIs for realistic tool interaction
2. **Domain-Specific Policy Documents:** Guidelines dictating required agent behavior for policy compliance testing
3. **LLM-Based User Simulator:** Guided by instructions for diverse scenarios to produce realistic user utterances

**Example Use Case:**
For an airline reservation agent, if a user wants to change their flight to a different destination, the agent must: (1) gather required information through user interaction, (2) check airline policies, (3) search for new flights using complex APIs, and (4) rebook if possible—all while following strict policy guidelines.

**Evaluation and Metrics:**

- **Stateful Evaluation:** Compares database state after task completion with expected outcome for objective measurement
- **pass^k Metric:** New reliability metric that measures if an agent can successfully complete the same task multiple times (k trials), ensuring consistency at scale
- **Goal-Based Assessment:** Focuses on database state rather than conversation quality, enabling fast and faithful evaluation without human or LLM judges

**Key Features:**

- **Realistic Dialog and Tool Use:** Complex databases and realistic LLM-powered user simulation with varied scenarios specified in natural language
- **Open-Ended and Diverse Tasks:** Rich data schemas, APIs, and policy documents support creative, diverse agent tasks
- **Faithful Objective Evaluation:** Database state evaluation provides fast, objective assessment of agent capabilities
- **Modular Framework:** Easy addition of new domains, database entries, rules, APIs, tasks, and evaluation metrics

**Findings:**

Results show that agents built with simple LLM constructs (like function calling or ReAct) **perform poorly on even relatively simple tasks**, highlighting the need for more sophisticated agent architectures. Frontier models often struggle to exceed **60-70% success rates** due to strict policy adherence requirements, demonstrating significant room for improvement in agent reliability.

---

### Vending-Bench 2

| Attribute | Details |
|-----------|---------|
| **Paper** | [Vending-Bench: A Benchmark for Long-Term Coherence](https://arxiv.org/abs/2502.15840) |
| **Website** | [andonlabs.com/evals/vending-bench-2](https://andonlabs.com/evals/vending-bench-2) |
| **Task** | Manage a vending machine business for a simulated year |
| **Starting Balance** | $500 |
| **Metric** | Bank account balance at year end |
| **Scale** | 3,000-6,000 messages, 60-100M tokens per run |
| **Status** | Active (V2) |

Vending-Bench 2 is a benchmark for measuring AI model performance on running a business over long time horizons. Models are tasked with running a simulated vending machine business over a year and scored on their bank account balance at the end.

**The Importance of Long-Term Coherence:**

Long-term coherence in agents is more important than ever. Coding agents can now write code autonomously for hours, and the length and breadth of tasks AI models can complete is increasing. Models are expected to soon take active part in the economy, managing entire businesses. But to do this, they must stay coherent and efficient over very long time horizons—this is what Vending-Bench 2 measures.

**Improvements from Original Vending-Bench:**

Vending-Bench 2 keeps the core business management concept but introduces more real-world messiness inspired by actual vending machine deployments:

- **Adversarial Suppliers:** Suppliers may actively try to exploit the agent, quoting unreasonable prices or using bait-and-switch tactics. Agents must recognize this and seek alternative options to stay profitable
- **Negotiation Required:** Even honest suppliers will try to maximize their profit. Negotiation is key to success
- **Supply Chain Disruptions:** Deliveries can be delayed and trusted suppliers can go out of business, forcing agents to build robust supply chains and always have a plan B
- **Customer Complaints:** Unhappy customers can reach out at any time demanding costly refunds
- **Streamlined Scoring:** Simplified evaluation based on money balance after a year with clarified optimization criteria
- **Better Planning Tools:** Proper note-taking and reminder systems added to support long-term planning

**How Vending-Bench Works:**

Models are given a **$500 starting balance** and must make as much money as possible managing their vending business over one year. Key mechanics:

**Bankruptcy and Termination:**
- Daily fee: $2 for the vending machine
- If the agent fails to pay for more than 10 consecutive days, they're terminated early

**Operations:**
- **Search the Internet:** Find suitable suppliers
- **Email Communication:** Contact suppliers to make orders
- **Inventory Management:** Move items between storage facility and vending machine using provided tools
- **Revenue Generation:** Dependent on factors like day of the week, season, weather, and pricing strategy

**Benchmark Scale:**

Running a model for a full year results in:
- **3,000-6,000 messages** in total
- **60-100 million tokens** in output during a run

This extensive scale tests whether models can maintain coherence, make strategic decisions, and adapt to changing circumstances over extremely long contexts.

**Findings:**

Results show that while models are improving at long-term coherence, current frontier models handle this challenge with **varying degrees of success**. The benchmark reveals significant room for improvement in maintaining strategic thinking and efficiency over extended time horizons.

---

### GAIA (General AI Assistants)

| Attribute | Details |
|-----------|---------|
| **Paper** | [GAIA: a benchmark for General AI Assistants](https://arxiv.org/abs/2311.12983) |
| **Website** | [Meta AI Research](https://ai.meta.com/research/publications/gaia-a-benchmark-for-general-ai-assistants/) |
| **Dataset** | [HuggingFace](https://huggingface.co/gaia-benchmark) |
| **Questions** | 466 real-world questions |
| **Human Performance** | 92% |
| **GPT-4 with Plugins** | 15% |

GAIA is a benchmark for General AI Assistants that, if solved, would represent a milestone in AI research. It proposes real-world questions that require a set of fundamental abilities such as reasoning, multi-modality handling, web browsing, and generally tool-use proficiency.

**The Human-AI Performance Gap:**

GAIA questions are **conceptually simple for humans yet challenging for most advanced AIs**:
- **Human respondents:** 92% success rate
- **GPT-4 equipped with plugins:** 15% success rate

This **notable performance disparity** contrasts sharply with the recent trend of LLMs outperforming humans on tasks requiring professional skills in areas like law or chemistry. While AI excels at specialized professional tasks, it struggles with questions that average humans find straightforward.

**GAIA's Philosophy:**

GAIA's philosophy departs from the current trend in AI benchmarks that target tasks ever more difficult for humans. Instead, GAIA posits that **the advent of Artificial General Intelligence (AGI) hinges on a system's capability to exhibit similar robustness as the average human does on such questions**.

The benchmark shifts focus from specialized expertise to general competence—the ability to handle everyday questions that require:
- **Reasoning:** Multi-step logical thinking
- **Multi-modality Handling:** Processing text, images, and other data types
- **Web Browsing:** Finding and synthesizing information online
- **Tool-Use Proficiency:** Effectively utilizing available tools and resources

**Dataset:**

Using GAIA's methodology, researchers devised **466 questions and their answers**, each designed to test the fundamental abilities required for general AI assistance rather than narrow domain expertise.

**Significance:**

GAIA represents a philosophical shift in how we evaluate progress toward AGI—measuring not how well AI can perform superhuman feats of expertise, but how robustly it can handle the conceptually simple tasks that any average human navigates with ease. This makes GAIA a critical benchmark for understanding the gap between narrow AI capabilities and true general intelligence.

---

## Emotional Intelligence

### EQ-Bench 3

| Attribute | Details |
|-----------|---------|
| **Paper** | [EQ-Bench: An Emotional Intelligence Benchmark](https://arxiv.org/abs/2312.06281) |
| **Website** | [eqbench.com](https://eqbench.com/) |
| **Scenarios** | 45 challenging roleplay scenarios |
| **Judge** | Claude Sonnet 3.7 (LLM-judged) |
| **Metrics** | Rubric scores & Elo ratings |
| **Elo Normalization** | o3=1500, llama-3.2-1b=200 |

EQ-Bench 3 is an LLM-judged benchmark that tests emotional intelligence (EQ) through challenging role-plays and analysis tasks. It measures empathy, social skills, and insight in scenarios like relationship conflicts and workplace dilemmas.

**Why EQ-Bench 3?**

Standard EQ tests are too easy for LLMs, and existing benchmarks often miss nuanced social skills crucial for human-AI interaction. EQ-Bench 3 uses difficult, free-form role-plays to better discriminate between models, addressing the gap between simple emotional understanding and the complex social intelligence needed for realistic interactions.

**How It Works:**

EQ-Bench 3 employs a structured evaluation process:

1. **In-Character Response:** Models respond as the character in the scenario
2. **Internal Reasoning:** Models explain their thought process ("I'm thinking/feeling...")
3. **Debriefing:** Models reflect on the interaction and their decisions

Responses are then evaluated through:
- **Detailed Rubric:** Judges responses against specific criteria
- **Pairwise Comparisons:** Models are compared head-to-head to generate Elo ratings
- **Elo Normalization:** Scores normalized with o3=1500 and llama-3.2-1b=200 as anchors

**Scoring Systems:**

**1. Rubric Scores (Absolute)**
- Evaluates responses against predefined criteria
- Less discriminative but provides clear benchmarks
- Useful for understanding specific strengths and weaknesses

**2. Elo Scores (Relative)**
- Based on pairwise comparisons between models
- More discriminative, revealing finer performance differences
- Better for ranking models and understanding relative capabilities

**Key Features:**

- **Active EQ Focus:** Tests practical application of emotional intelligence in realistic scenarios, not just theoretical understanding
- **Challenging & Discriminative Scenarios:** Difficult role-plays (relationship conflicts, workplace dilemmas) that effectively differentiate model capabilities
- **Bias Mitigation:**
  - **Length Truncation:** Applied to Elo comparisons to reduce length bias
  - **Position Bias Control:** Ensures fair evaluation regardless of response ordering
- **Full Transcripts:** Complete interaction transcripts available for detailed analysis and transparency

EQ-Bench 3 provides a rigorous evaluation of emotional intelligence that goes beyond simple sentiment analysis, testing models on the nuanced social skills essential for natural human-AI interaction.

---

## Factuality and Truthfulness

### SimpleQA / SimpleQA Verified

| Attribute | Details |
|-----------|---------|
| **Paper** | [SimpleQA Verified](https://arxiv.org/abs/2509.07968) |
| **Website** | [OpenAI](https://openai.com/index/introducing-simpleqa/) |
| **Created By** | OpenAI |
| **Questions** | 4,326 short, fact-seeking questions |
| **Focus** | Short-form factual accuracy (hallucination rate) |

| **Estimated Error Rate** | ~3% |

SimpleQA is a factuality benchmark that measures the ability of language models to answer short, fact-seeking questions. It addresses a critical open problem in AI: how to train models that produce factually correct responses and avoid "hallucinations"—false outputs or answers unsubstantiated by evidence.

**The Hallucination Problem:**

Current language models sometimes produce false outputs or answers unsubstantiated by evidence. Language models that generate more accurate responses with fewer hallucinations are more trustworthy and can be used in a broader range of applications. Measuring factuality is challenging because evaluating arbitrary factual claims is difficult, and language models can generate long completions containing dozens of claims. SimpleQA focuses on **short, fact-seeking queries**, reducing scope but making factuality measurement much more tractable.

**Four Key Design Goals:**

**1. High Correctness**
- Reference answers supported by sources from **two independent AI trainers**
- Questions written so predicted answers are easy to grade
- Quality verification through multiple stages

**2. Diversity**
- Covers a wide range of topics: science, technology, TV shows, video games, and more
- Ensures broad evaluation across different knowledge domains

**3. Challenging for Frontier Models**
- Compared to older benchmarks (TriviaQA 2017, NQ 2019), which have become saturated
- Created to challenge frontier models: **GPT-4o scores less than 40%**
- Tests the limits of current model capabilities

**4. Good Researcher UX**
- Fast and simple to run with concise questions and answers
- Efficient grading through OpenAI API or other frontier model APIs
- **4,326 questions** provide relatively low variance as an evaluation benchmark

**Creation Methodology:**

OpenAI hired AI trainers to browse the web and create questions and answers. To be included in the dataset, each question had to meet strict criteria:

- **Single, Indisputable Answer:** For easy grading
- **Time-Invariant:** Answer should not change over time
- **Hallucination-Inducing:** Most questions had to induce hallucinations from either GPT-4o or GPT-3.5

To ensure quality, a **second, independent AI trainer** answered each question without seeing the original response. Only questions where **both trainers' answers agreed** were included.

**Quality Verification:**

OpenAI conducted rigorous quality verification:

- A **third AI trainer** answered a random sample of 1,000 questions
- **94.4% agreement rate** with original answers (5.6% disagreement)
- Manual inspection of disagreements revealed:
  - **2.8% due to grader errors:** False negatives or human errors from the third trainer (incomplete answers, misinterpreting sources)
  - **2.8% due to real issues:** Ambiguous questions or conflicting website answers
- **Estimated inherent error rate: ~3%**

**Focus and Scope:**

SimpleQA measures LLMs' **parametric factuality**—their ability to correctly answer short factual questions from their training knowledge (not retrieval). By focusing on tractable, well-defined questions with verifiable answers, SimpleQA provides a reliable measure of model factuality while avoiding the complexity of evaluating long-form, multi-claim responses.

---

## Commonsense Reasoning

### HellaSwag

| Attribute | Details |
|-----------|---------|
| **Paper** | [HellaSwag: Can a Machine Really Finish Your Sentence?](https://arxiv.org/abs/2502.11393) |
| **Introduced** | 2019 (Zellers et al.) |
| **Validation Set** | 10,000+ tasks |
| **Sources** | ActivityNet, WikiHow (video caption datasets) |
| **Format** | Multiple-choice sentence completion (1 correct + 3 adversarial) |
| **Human Performance** | 95.6% |
| **Open Models** | ~80% |
| **Top Proprietary Models** | ~90% |

**HellaSwag** stands for **Harder Endings, Longer contexts, Low-shot Activities for Situations With Adversarial Generations**. It remains one of the most relevant benchmarks in 2025 for measuring commonsense reasoning in large language models. 

**Performance Gap and Relevance:**

While humans reach **95.6% accuracy**, most open models stay around **80%** and only the strongest proprietary models approach **90%**. This persistent gap reveals why HellaSwag remains critical: it tests subtle understanding of everyday actions that expose fundamental limitations in current AI systems.

**What Makes HellaSwag Hard:**

Unlike trivia or math tasks, **commonsense is invisible**. Models often miss it. The benchmark tests understanding of:
- **Time sequence:** What logically happens next
- **Physical laws:** How objects and bodies work in the real world
- **Social norms:** Expected human behavior patterns

That gap reveals why true understanding remains elusive for AI. HellaSwag highlights this disconnect and has become a **diagnostic tool for the limits of current language systems**.

**How It Works:**

Each item in the dataset starts with a short description (1-2 sentences) taken from video caption datasets like **ActivityNet** or **WikiHow**. The model must pick the correct ending from four options:

1. **One correct ending:** Written by a human
2. **Three adversarial endings:** Designed to be misleading

This is not random. These wrong endings are generated using **adversarial filtering**—chosen specifically because weaker models were fooled by them. The process:
- Sample completions from language models
- Keep only those that confuse machines but not humans
- Ensure they are grammatically correct and superficially plausible
- But don't match real-world logic

**Dataset Composition:**

The full validation set includes over **10,000 tasks**. Many involve everyday human actions like:
- Opening a fridge
- Walking through a doorway
- Cooking
- Interacting with objects

This forces the model to reason about what should happen next in simple, everyday scenarios—areas where large language models still fall short.

**Natural Language Inference:**

HellaSwag is also a test of natural language inference. The system needs to complete a story based on **what is implied, not only what is said**. That shift makes HellaSwag more robust than older benchmarks like ARC, requiring models to demonstrate genuine understanding of physical and social reality rather than pattern matching.

---

## Frontier and Expert-Level

### Humanity's Last Exam (HLE)

| Attribute | Details |
|-----------|---------|
| **Paper** | [Humanity's Last Exam](https://arxiv.org/abs/2501.14249) |
| **Website** | [agi.safe.ai](https://agi.safe.ai/) |
| **Dataset** | [HuggingFace](https://huggingface.co/datasets/cais/hle) |
| **Questions** | 2,500-3,000 across 100+ disciplines |
| **Public Set** | 2,500 finalized + private test set |
| **Format** | Closed-ended (multiple-choice, exact-match short answer) |
| **Multi-modal** | ~14% require images or diagrams |
| **Contributors** | Nearly 1,000 experts from 500+ institutions |
| **Prize Pool** | $500,000 for high-quality questions |
| **AI Performance** | <30% (initial release) |
| **Human Graduates** | ~90% |
| **Status** | Active, extremely hard |

Humanity's Last Exam (HLE) is a challenging, multi-modal benchmark designed to evaluate advanced language models on expert-level reasoning and knowledge, where current AI models significantly underperform human experts. Designed to be **"the final closed-ended academic benchmark,"** it contains extremely challenging questions at the frontier of human knowledge.

**Purpose and Motivation:**

HLE was created to address the problem of **benchmark saturation**, where previous tests like MMLU had become too easy for state-of-the-art AI. The benchmark aims to measure **genuine reasoning capabilities and expert-level understanding**, rather than simple information retrieval or pattern recognition.

**Content and Format:**

The benchmark consists of approximately **2,500-3,000 questions** across more than **100 academic disciplines**, including:
- Mathematics
- Humanities
- Natural sciences
- And many other expert-level domains

**Question Characteristics:**

- **Closed-Ended:** Multiple-choice or exact-match short answer format
- **Multi-Modal:** About **14% require interpretation of images or diagrams**
- **Extremely Difficult:** Filtered to ensure frontier LLMs could not answer them correctly at creation time
- **Non-Searchable:** Original or non-trivial syntheses of information, making it impossible for AI to succeed through simple web searches or memorization of online facts

**Collaborative Development:**

HLE was developed as a **global, collaborative effort**:
- Nearly **1,000 subject-matter experts** (professors and researchers)
- From **500+ institutions**
- **$500,000 prize pool** to incentivize high-quality, challenging questions

This collaborative approach ensures diverse expertise and rigorous quality standards across all academic disciplines.

**Performance and the AI-Human Gap:**

As of its initial release, the results reveal a significant gap in reasoning abilities:
- **Best AI models:** Below **30%** accuracy
- **Human graduate students:** Nearly **90%** accuracy

This substantial performance gap highlights fundamental limitations in current AI reasoning capabilities when faced with expert-level questions requiring deep understanding.

**Calibration Issues:**

Models also tend to show **high confidence in their incorrect answers**, a phenomenon known as **poor calibration**. This means AI systems not only fail to answer correctly but also fail to recognize their own uncertainty—a critical problem for deploying AI in high-stakes domains.

**Impact and Use Cases:**

HLE serves as a key tool for:
- **Tracking AI Progress:** Measuring improvements in reasoning and expert-level understanding over time
- **Informing Policy:** Providing data for regulatory discussions about AI capabilities and limitations
- **Research Benchmarking:** Enabling rigorous evaluation of new model architectures and training approaches

**Quality Assurance and Revisions:**

While HLE represents a significant benchmark achievement, it has faced some criticism. Some analyses suggest a notable percentage of chemistry/biology questions may be:
- Ambiguous
- Have answers that conflict with some peer-reviewed literature

In response, the creators plan a **rolling revision process** to continuously improve question quality and address identified issues, ensuring the benchmark remains rigorous and fair.

---

### GDPval (GDP-value Evaluation)

| Attribute | Details |
|-----------|---------|
| **Paper** | [GDPval: Evaluating AI Model Performance on Real-World Economically Valuable Tasks](https://arxiv.org/abs/2510.04374) |
| **Website** | [OpenAI (GDP-value Evaluation)](https://openai.com/index/gdpval/) |
| **Created By** | OpenAI |
| **Tasks** | 1,320 (full set), 220 (gold subset) |
| **Occupations** | 44 across 9 U.S. GDP sectors |
| **Expert Experience** | Average 14 years professional experience |
| **Task Duration** | Average 7 hours (up to multiple weeks) |
| **Metric** | Win rate (pairwise human expert comparison) |
| **Status** | Active, open-sourced gold subset |

GDPval is a benchmark designed to evaluate AI model performance on real-world, economically valuable professional tasks. Unlike academic-style benchmarks that focus on reasoning difficulty, GDPval measures AI capabilities on the actual deliverables that professionals create daily—presentations, spreadsheets, reports, legal briefs, engineering blueprints, patient care plans, and financial analyses.

**Motivation and Philosophy:**

Current AI benchmarks often focus on abstract tests and academic challenges. However, to understand AI's potential economic impact, we need to measure performance on tasks that directly contribute to economic value. GDPval addresses this by:

- **Measuring Real Work:** Tasks are based on actual work product from industry experts, not synthetic test problems
- **Economic Representativeness:** Covers the top 9 sectors contributing to U.S. GDP, with 44 occupations that collectively earn $3 trillion annually
- **Professional Validation:** All tasks are constructed and graded by industry professionals with an average of 14 years of experience

**Benchmark Coverage:**

GDPval covers occupations across 9 major U.S. GDP-contributing sectors (each contributing >5% to GDP):
- Software Engineers
- Nurses and Healthcare Professionals
- Lawyers and Legal Analysts
- Financial Analysts and Accountants
- Journalists and Content Creators
- Engineers (various disciplines)
- Business Analysts and Consultants
- Marketing and Sales Professionals
- And more across education, government, and other sectors

**Task Characteristics:**

**Multi-Modal and Complex Deliverables:**
- Tasks require manipulating various formats: CAD design files, photos, videos, audio, social media posts, diagrams, slide decks, spreadsheets, and customer support conversations
- Each task may require parsing through up to 17 reference files (gold subset) or 38 files (full set)
- Deliverables span documents, slides, diagrams, spreadsheets, and multimedia

**Long-Horizon Difficulty:**
- Tasks require an **average of 7 hours** of work for an expert professional to complete
- On the high end, tasks span **up to multiple weeks** of work
- This tests sustained reasoning and coherence over extended work periods

**Evaluation Methodology:**

**Human Expert Grading (Primary Method):**
- Blinded pairwise comparisons by industry professionals
- Experts in the relevant occupation rank unlabeled work deliverables
- Grading each comparison takes over an hour on average
- Experts provide detailed justifications for rankings

**Automated Grader (Experimental):**
- Available at [evals.openai.com](https://evals.openai.com)
- Achieves **66% agreement** with human expert graders
- Human expert inter-rater agreement: **71%**
- Faster and cheaper alternative, but not a full substitute for expert grading

**Key Findings:**

**Model Performance:**
- **GPT-5.2 Thinking** achieved a **70.9% win rate**, surpassing human expert work quality
- Claude Opus 4.1 achieved **47.6% of deliverables** graded as better than or equal to human expert work
- Claude excels on **aesthetics** (document formatting, slide layout)
- GPT-5 excels on **accuracy** (instruction following, correct calculations)
- Model deliverables outperformed or matched expert humans' deliverables in just over half the tasks

**Speed and Cost Analysis:**
- Frontier AI models can potentially complete tasks **significantly faster and cheaper** than unaided experts
- Under a "try AI first, fix if needed" workflow, substantial time and cost savings are possible
- This enables analysis of ROI for AI integration in professional workflows

**Unique Advantages:**

1. **Realism:** Tasks based on actual professional work, not academic exercises
2. **Representative Breadth:** 1,320 tasks across 44 occupations covering majority of O*NET Work Activities
3. **Subjectivity Evaluation:** Graders consider style, format, aesthetics, and relevance—not just correctness
4. **No Upper Limit:** Win rate metric allows continuous evaluation without saturation
5. **Computer Use and Multi-Modality:** Tests real-world tool and format manipulation

**Limitations:**

- Focus on self-contained digital knowledge work (excludes manual labor, physical tasks)
- Tasks are precisely-specified and one-shot, not fully interactive
- Does not capture tasks requiring extensive tacit knowledge, PII access, or proprietary software
- Current automated grader has limitations compared to human expert evaluation

**Open-Source Availability:**

OpenAI has open-sourced the **220-task gold subset**, including prompts and reference files, to facilitate research in understanding real-world AI capabilities.

---

## Leaderboards and Resources

### Major Leaderboards

| Leaderboard | Focus | URL |
|-------------|-------|-----|
| **LMArena (Chatbot Arena)** | Human preference (5M+ votes) | [lmarena.ai](https://lmarena.ai/) |
| **Vellum LLM Leaderboard** | Multi-benchmark (excludes saturated benchmarks) | [vellum.ai/llm-leaderboard](https://www.vellum.ai/llm-leaderboard) |
| **Open LLM Leaderboard** | Open-source models | [vellum.ai/open-llm-leaderboard](https://www.vellum.ai/open-llm-leaderboard) |
| **SEAL Leaderboards** | Expert-driven (HLE, SWE-bench Pro) | [scale.com/leaderboard](https://scale.com/leaderboard) |
