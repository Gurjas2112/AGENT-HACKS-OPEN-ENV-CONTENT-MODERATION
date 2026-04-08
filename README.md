# AGENT-HACKS-OPEN-ENV-CUSTOMER-SERVICE-AGENT

Complex environment where agents resolve multi-step queries using external tools and APIs. This OpenEnv environment adapts devpulse_ai for code review tasks with three difficulty levels: easy (relevance scoring), medium (risk assessment), and hard (synthesis).

## Overview

This environment simulates code review intelligence gathering using multi-agent pipelines from devpulse_ai. Agents process GitHub signals (PRs/issues) to score relevance, assess risks, and synthesize insights.

## Tasks

- **Easy**: Score relevance of at least 3 signals (incremental rewards).
- **Medium**: Assess risks on at least 5 signals (penalize loops).
- **Hard**: Complete synthesis with prior actions (full pipeline).

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Install dependencies: `pip install -r envs/devpulse_code_review_env/server/requirements.txt`
3. **Set environment variables in `.env` (HF_TOKEN as well as OPENAI_API_KEY required)**:
   - Copy `.env.example` to `.env`.
   - Get HF_TOKEN from [Hugging Face Tokens](https://huggingface.co/settings/tokens).
   - Get OPENAI_API_KEY from [OpenAI API Keys](https://platform.openai.com/api-keys).
4. Run server: `cd envs/devpulse_code_review_env/server && uvicorn app:app --host 0.0.0.0 --port 8000`
5. Test inference: `python inference.py easy`

## Inference

The `inference.py` script uses OpenAI API for baseline LLM-driven actions. Output format matches hackathon requirements.

## Deployment

- Docker: `docker build -t openenv-devpulse -f envs/devpulse_code_review_env/server/Dockerfile`
- HF Spaces: Push to GitHub, create Space with Docker SDK, tag as `openenv`.

## Baseline Performance

- Heuristic: 0.5
- LLM (gpt-4o-mini): 0.8+

## Files

- `envs/devpulse_code_review_env/`: Environment code (models, client, server).
- `devpulse_ai/`: Original multi-agent app.
- `inference.py`: Hackathon inference script.
- `openenv.yaml`: Environment metadata.

## Testing/Validation

- Install openenv-core: `pip install openenv-core`
- Run validate: `openenv validate` (from repo root)
- Run validator script: Download and run `./validate-submission.sh https://your-space.hf.space`.
- For submission: Update `HF_SPACE_URL` in `inference.py` and `env` vars.

## License

MIT License.