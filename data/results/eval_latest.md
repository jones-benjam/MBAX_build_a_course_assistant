# Evaluation results (2026-10-15T20:58:53+00:00)

Files: MBAX 6418 Syllabus Fall 2026.pdf, MBAX 6418 - Week 2 - LLM Fundamentals v2.pptx, MBAX 6418 - Week 3 - Prompt Engineering v1.pptx, MBAX 6418 - Week 4 - Serving and Debugging v6.pptx, MBAX 6418 - Week 5 - Context Engineering and RAG v1.pptx  
Questions: 10, top_k=8, services: class endpoints

## Reranker ON vs OFF

| Metric | Rerank ON | Rerank OFF |
|---|---|---|
| Correct answers | 8/10 | 9/10 |
| Answers passing the support check | 10/10 | 10/10 |
| Expected page cited (answerable qs) | 9 | 9 |
| Expected page in retrieved top-k | 9 | 9 |
| Mean rank of expected page | 1.0 | 1.67 |
| Mean retrieval latency (s) | 3.92 | 0.44 |
| Mean end-to-end latency (s) | 5.09 | 1.88 |

## Per question

| Id | Type | Rerank | Correct | Keywords | Expected page cited | Supported | Image | Rank | Retrieval s | Answer s |
|---|---|---|---|---|---|---|---|---|---|---|
| syl-grading | syllabus | on | ✅ | 4/4 | True | True | True | 1 | 3.89 | 4.18 |
| syl-office-hours | syllabus | on | ✅ | 4/4 | True | True | True | 1 | 6.94 | 4.93 |
| syl-late-policy | syllabus | on | ❌ | 3/4 | True | True | True | 1 | 3.17 | 4.28 |
| w2-quantization | slide-text | on | ✅ | 4/4 | True | True | True | 1 | 4.02 | 3.4 |
| w3-chatml | slide-text | on | ❌ | 1/3 | True | True | True | 1 | 3.23 | 4.16 |
| w4-branching | slide-text | on | ✅ | 3/3 | True | True | True | 1 | 4.58 | 14.78 |
| w5-reranking | slide-text | on | ✅ | 3/3 | True | True | True | 1 | 3.72 | 3.27 |
| w2-meme-visual | visual | on | ✅ | 3/3 | True | True | True | 1 | 3.69 | 4.65 |
| w2-trends-visual | visual | on | ✅ | 3/3 | True | True | True | 1 | 3.17 | 4.36 |
| unanswerable | unanswerable | on | ✅ | n/a | None | True | False | None | 2.76 | 2.89 |
| syl-grading | syllabus | off | ✅ | 4/4 | True | True | True | 2 | 0.4 | 2.51 |
| syl-office-hours | syllabus | off | ✅ | 4/4 | True | True | True | 1 | 0.48 | 1.68 |
| syl-late-policy | syllabus | off | ✅ | 4/4 | True | True | True | 1 | 0.42 | 1.89 |
| w2-quantization | slide-text | off | ✅ | 4/4 | True | True | True | 1 | 0.48 | 1.77 |
| w3-chatml | slide-text | off | ❌ | 1/3 | True | True | True | 3 | 0.39 | 1.59 |
| w4-branching | slide-text | off | ✅ | 3/3 | True | True | True | 1 | 0.46 | 1.58 |
| w5-reranking | slide-text | off | ✅ | 3/3 | True | True | True | 1 | 0.39 | 1.55 |
| w2-meme-visual | visual | off | ✅ | 3/3 | True | True | True | 3 | 0.38 | 2.35 |
| w2-trends-visual | visual | off | ✅ | 3/3 | True | True | True | 2 | 0.56 | 2.23 |
| unanswerable | unanswerable | off | ✅ | n/a | None | True | False | None | 0.4 | 1.61 |

## Answers

### syl-grading (rerank on)

**Q:** How is the final grade calculated in this course?

**A:** The final grade is calculated based on the following components: Assignments (20%), Quizzes (15%), Final Project (45%), Final Exam (15%), and Attendance and Participation (5%) [S1]. The instructor reserves the right to grade on the curve [S1].

Sources: MBAX 6418 Syllabus Fall 2026.pdf p.2

### syl-office-hours (rerank on)

**Q:** When are the professor's office hours and how are they held?

**A:** The professor's office hours are on Tuesdays and Fridays from 11am to 12pm, and they are held via Zoom.

Sources: MBAX 6418 Syllabus Fall 2026.pdf p.1

### syl-late-policy (rerank on)

**Q:** What is the late policy for assignments and quizzes?

**A:** For assignments, there is a 10% grade penalty for every calendar day the assignment is late, and assignments may not be turned in more than five days late [S5], [S1]. For quizzes, the instructor will NOT accept late quiz submissions [S2].

Sources: MBAX 6418 Syllabus Fall 2026.pdf p.2, MBAX 6418 Syllabus Fall 2026.pdf p.3, MBAX 6418 Syllabus Fall 2026.pdf p.3

### w2-quantization (rerank on)

**Q:** What is quantization, and roughly how large is Qwen3-Coder-30B-A3B-Instruct at 16, 8 and 4 bits?

**A:** Quantization is a process for scaling and rounding model parameters to make models smaller and potentially faster [S1]. For the Qwen3-Coder-30B-A3B-Instruct model, the approximate sizes are: 16-bits: ~61GB; 8-bits: ~31GB; 4-bits: ~17GB [S1].

Sources: MBAX 6418 - Week 2 - LLM Fundamentals v2.pptx p.15

### w3-chatml (rerank on)

**Q:** What is ChatML and which special tokens does it use to mark messages?

**A:** ChatML (Chat Markup Language) is a format used for structuring and presenting content to a model, as part of prompt formatting/templates [S3]. According to the provided evidence, ChatML uses the special tokens `<|im_start|>` to mark the beginning of a message and `` to mark the end of a message [S1, S2, S7].

Sources: MBAX 6418 - Week 3 - Prompt Engineering v1.pptx p.8, MBAX 6418 - Week 3 - Prompt Engineering v1.pptx p.10, MBAX 6418 - Week 3 - Prompt Engineering v1.pptx p.7

Warnings: Citation [S7] looked like a page/slide number; it was mapped to [S3] (MBAX 6418 - Week 3 - Prompt Engineering v1.pptx, page/slide 7), which supports the text.

### w4-branching (rerank on)

**Q:** According to the Week 4 slides, why would you create a branch in a repository, and what is the default branch usually called?

**A:** According to the Week 4 slides, you would create a branch to "create separate lines of work in a repo to safely" develop a new feature, fix a bug, or experiment. The default branch is typically called "main".

Sources: MBAX 6418 - Week 4 - Serving and Debugging v6.pptx p.25

### w5-reranking (rerank on)

**Q:** In a hybrid RAG pipeline, what does the reranking step do?

**A:** In a hybrid RAG pipeline, the reranking step "compares these candidates to the original question and prioritizes which chunks are supplied to the LLM as context" [S1]. The diagram in the slide shows that after retrieval produces "Candidate Chunks," the "Reranker" processes them to output "Ranked Chunks With Source Details" [S1].

Sources: MBAX 6418 - Week 5 - Context Engineering and RAG v1.pptx p.18

### w2-meme-visual (rerank on)

**Q:** Find the meme about Vibe Coding on Prod in the Week 2 slides and explain what its image and text show.

**A:** The meme about Vibe Coding on Prod is found on slide 33 [S1]. The image shows the character Boromir from "The Lord of the Rings" with the text "ONE DOES NOT SIMPLY" at the top and "VIBE CODE A PRODUCTION-GRADE ENTERPRISE APP" at the bottom. The slide title is "Vibe Coding on 'Prod'" [S1].

Sources: MBAX 6418 - Week 2 - LLM Fundamentals v2.pptx p.33

### w2-trends-visual (rerank on)

**Q:** What does the chart on the 'Vibe Coding' slide in Week 2 show, and what happened around 2025?

**A:** The chart on the 'Vibe Coding' slide (slide 29) shows a Google Trends graph titled "Interest Over Time for 'Vibe Coding' and 'Learn to Code' on Google in the US from July 2024" [S1]. The chart shows that around January 2025, there was a massive spike in searches for "Vibe Coding" (the green line) that surpassed searches for "Learn to Code" (the dark green line) [S1]. This spike is attributed to a tweet by Andrej Karpathy, which is also shown on the slide [S1].

Sources: MBAX 6418 - Week 2 - LLM Fundamentals v2.pptx p.29

### unanswerable (rerank on)

**Q:** What was the class average score on Assignment 1?

**A:** The provided evidence does not contain the class average score on Assignment 1.

Sources: none

### syl-grading (rerank off)

**Q:** How is the final grade calculated in this course?

**A:** The final grade is calculated based on the following weighted components [S2]:

*   **Final Project (45%)**: A team-based project involving a proposal and a brief presentation [S4].
*   **Assignments (20%)**: Individual or team-based assignments completed throughout the semester [S5]. The lowest assignment score is dropped from the final grade [S1].
*   **Final Exam (15%)**: A remotely completed exam similar in content to class assignments [S4].
*   **Quizzes (15%)**: Quizzes given during class or on Canvas; the lowest quiz score is dropped from the final grade [S3].
*   **Attendance and Participation (5%)**: Based on attendance and positive contributions to class discussions [S6].

Sources: MBAX 6418 Syllabus Fall 2026.pdf p.3, MBAX 6418 Syllabus Fall 2026.pdf p.2, MBAX 6418 Syllabus Fall 2026.pdf p.3, MBAX 6418 Syllabus Fall 2026.pdf p.3, MBAX 6418 Syllabus Fall 2026.pdf p.2, MBAX 6418 Syllabus Fall 2026.pdf p.7

### syl-office-hours (rerank off)

**Q:** When are the professor's office hours and how are they held?

**A:** The professor's office hours are held on Tuesdays and Fridays from 11am to 12pm via Zoom. They are also available by appointment.

Sources: MBAX 6418 Syllabus Fall 2026.pdf p.1

Warnings: Model cited non-existent evidence ids [10]; they were removed.

### syl-late-policy (rerank off)

**Q:** What is the late policy for assignments and quizzes?

**A:** For assignments, there is a 10% grade penalty for every calendar day the assignment is late. Assignments may not be turned in more than five days late, but the lowest score will be dropped from the final grade [S1, S3]. For quizzes, the instructor will NOT accept late submissions, but the quiz with the lowest score will be dropped from the final grade [S2].

Sources: MBAX 6418 Syllabus Fall 2026.pdf p.3, MBAX 6418 Syllabus Fall 2026.pdf p.3, MBAX 6418 Syllabus Fall 2026.pdf p.2

### w2-quantization (rerank off)

**Q:** What is quantization, and roughly how large is Qwen3-Coder-30B-A3B-Instruct at 16, 8 and 4 bits?

**A:** Quantization is a process for scaling and rounding model parameters to make models smaller and potentially faster [S1]. For the Qwen3-Coder-30B-A3B-Instruct model, the approximate sizes at different bit precisions are: 16-bits: ~61GB; 8-bits: ~31GB; and 4-bits: ~17GB [S1].

Sources: MBAX 6418 - Week 2 - LLM Fundamentals v2.pptx p.15

### w3-chatml (rerank off)

**Q:** What is ChatML and which special tokens does it use to mark messages?

**A:** ChatML (Chat Markup Language) is a model prompt formatting template that defines how content is structured and presented to the model. According to the evidence, it uses the special tokens `<|im_start|>` to mark the beginning of a message and `` to mark the end of a message.

Sources: MBAX 6418 - Week 3 - Prompt Engineering v1.pptx p.7, MBAX 6418 - Week 3 - Prompt Engineering v1.pptx p.8

### w4-branching (rerank off)

**Q:** According to the Week 4 slides, why would you create a branch in a repository, and what is the default branch usually called?

**A:** According to the Week 4 slides, you would create a branch to "create separate lines of work in a repo to safely" develop a new feature, fix a bug, or experiment. The default branch is typically called "main".

Sources: MBAX 6418 - Week 4 - Serving and Debugging v6.pptx p.25

### w5-reranking (rerank off)

**Q:** In a hybrid RAG pipeline, what does the reranking step do?

**A:** In a hybrid RAG pipeline, the reranking step "compares these candidates to the original question and prioritizes which chunks are supplied to the LLM as context" [S1].

Sources: MBAX 6418 - Week 5 - Context Engineering and RAG v1.pptx p.18

### w2-meme-visual (rerank off)

**Q:** Find the meme about Vibe Coding on Prod in the Week 2 slides and explain what its image and text show.

**A:** The meme about Vibe Coding on Prod is found on slide 33 [S3]. The image shows the popular "One Does Not Simply" meme featuring the character Boromir from The Lord of the Rings. The text overlaid on the image reads: "ONE DOES NOT SIMPLY VIBE CODE A PRODUCTION-GRADE ENTERPRISE APP".

Sources: MBAX 6418 - Week 2 - LLM Fundamentals v2.pptx p.33

### w2-trends-visual (rerank off)

**Q:** What does the chart on the 'Vibe Coding' slide in Week 2 show, and what happened around 2025?

**A:** The chart on the 'Vibe Coding' slide (slide 29) shows a Google Trends graph titled "Interest Over Time for 'Vibe Coding' and 'Learn to Code' on Google in the US from July 2024". It compares the search interest for the two terms. Around 2025, specifically after "Karpathy's tweet" (dated Feb 2, 2025), the search interest for "Vibe Coding" spiked dramatically, surpassing the search interest for "Learn to Code".

Sources: MBAX 6418 - Week 2 - LLM Fundamentals v2.pptx p.29

### unanswerable (rerank off)

**Q:** What was the class average score on Assignment 1?

**A:** The provided evidence does not contain the class average score on Assignment 1.

Sources: none
