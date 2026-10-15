# Class endpoints: verified request formats

Checked on 2026-10-15 against each server's `/openapi.json` (all five are vLLM
OpenAI-compatible servers; `/docs` is public, every data route needs
`Authorization: Bearer <key>`) and against the Hugging Face model cards. The
client payloads in `course_assistant/services/http_clients.py` were validated
offline against the downloaded request schemas; `tests/test_http_clients.py` pins
the shapes. Live behaviour (actual scores, JSON compliance) is covered by the
opt-in `tests/test_live_services.py`.

Base URLs are configured as `http://<host>:<port>` (the class list gives them
without `/v1`); the client appends `/v1` when it is missing. Dummy values live in
`.env.example`; real ones only in the gitignored `.env`.

| Port | Model | Route used | Request shape | Notes |
|------|-------|------------|---------------|-------|
| 9001 | `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` (vision LLM) | `POST /v1/chat/completions` | OpenAI messages; images as `{"type":"image_url","image_url":{"url":"data:image/png;base64,…"}}`; `response_format: {"type":"json_object"}`; `chat_template_kwargs: {"enable_thinking": false}`; `max_completion_tokens` | Any `<think>…</think>` block is stripped from the reply as a fallback. Port 9000 is reserved for another service. |
| 9002 | `nvidia/Nemotron-3-Embed-1B-BF16` (text embeddings) | `POST /v1/embeddings` | `{"model", "input": [str, …]}` | Model card: on this route callers must prefix `query: ` / `passage: ` themselves. The client does this (`embed` = passages, `embed_query` = queries). 2048-dim, L2-normalised. |
| 9003 | `Qwen/Qwen3-VL-Embedding-2B` (visual embeddings) | `POST /v1/embeddings` (chat form) | `{"model", "messages": [{"role":"system","content": instruction}, {"role":"user","content": [image_url part or text part]}]}` | The completion form (`input`) only accepts strings, so images use the chat form. One conversation per request. Default instruction for slides is "Represent the user's input."; queries use "Retrieve images or text relevant to the user's query." |
| 9004 | `Qwen/Qwen3-VL-Reranker-2B` (reranker) | `POST /v1/rerank` | `{"model", "query": str, "documents": [str or {"content": [typed parts]}], "top_n": n, "instruction": str}` | Response: `results[{index, relevance_score, document}]`. Candidates with an image send text + `image_url` parts. |
| 9005 | `dots.mocr` (document parser) | not used | chat-completions style OCR server | Local PyMuPDF + LibreOffice parsing is the implemented path. |

Error handling: any HTTP error, timeout or malformed body becomes
`ServiceUnavailable` with the status code and the server's short error message
(never the key or the request body). Retrieval degrades channel by channel; the
answerer and quiz generator fall back to showing the top passages.
