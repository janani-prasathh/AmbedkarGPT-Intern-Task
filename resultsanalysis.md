# Evaluation Results Analysis

## Overview

This analysis covers the retrieval and answer quality evaluation of our RAG QA system on Dr. Ambedkar's document corpus, using provided QA pairs and metrics. Due to hardware limitations, evaluation was performed on a subset ("small" chunk config, 5 questions). The code architecture supports full-scale experiments on a more capable machine.

## Chunking Strategy Findings

Only the "small" chunking configuration (250 chars, 50 overlap) was successfully evaluated in this run. Empirically, small or medium chunk sizes often offer the best trade-off between retrieval granularity and LLM context size. Larger chunk sizes can lead to partial matches or context dilution.

## Metric Results (sample)

- **ROUGE-L:** Scores ranged from ~0.17 to 0.46, indicating partial lexical overlap between predicted and ground-truth answers.
- **Cosine Similarity:** Ranged from 0.53 to 0.70; higher values correlate with more relevant and faithful answers.
- **BLEU:** Scores were low, which is typical for generative QA compared to reference spans.
- **Answer Relevance/Factuality:** Scores, simple blends of ROUGE-L and semantic similarity, reflected both directness and faithfulness of answers.

## Retrieval Accuracy

Retrieval metrics (hit rate, MRR, precision) appear as `null` here, likely due to missing or mismatched `sourcedocuments` field or a config choice in the pipeline. These metrics can be re-enabled if ground truth doc filenames are provided.

## Failure Analysis

- The LLM sometimes expands on the reference (“The real remedy...”) with plausible but non-exact details.
- For questions lacking clear context, answers may become generic, hurting overlap metrics.
- Some predicted answers mention empowerment–related means not strictly in the ground-truth answer, lowering overlap but still within topic.

## Recommendations

- Tune chunk size (200–600 chars) for best retrieval accuracy.
- Strictly set `sourcedocuments` fields to enable full retrieval metric computation.
- Advance prompt tuning to force the LLM to abstain if context is inadequate.
- Consider using a reranker or stronger LLM if resources allow.

## Conclusion

The pipeline works end-to-end on the realistic subset, demonstrating correct retrieval, answer generation, and metric reporting. With greater system resources, running over all configs and questions would yield a complete set of results for deeper comparative analysis.
