import asyncio
import os
from typing import List
from openai import OpenAI
from envs.devpulse_code_review_env import DevPulseAction, DevPulseCodeReviewEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
TASK_NAME = os.getenv("TASK", "easy")  # Default task
BENCHMARK = "devpulse_code_review"
MAX_STEPS = 10
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.5  # Normalized score in [0, 1]

# For submission, set to your HF Space URL (e.g., "https://your-space.hf.space")
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "http://localhost:8000")  # Change for submission

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str):
    error_str = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_str}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def get_model_action(step: int, task: str, signals: List[dict], completed_actions: List[str]) -> str:
    prompt = f"Task: {task}. Signals: {[s['title'] for s in signals]}. Completed: {completed_actions}. Choose: score_relevance (with index), assess_risk (with index), synthesize."
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = response.choices[0].message.content.strip()
        if "score" in text.lower():
            return "score_relevance"
        elif "assess" in text.lower():
            return "assess_risk"
        else:
            return "synthesize"
    except Exception as e:
        return "synthesize"  # Fallback

async def main():
    env = DevPulseCodeReviewEnv(base_url=HF_SPACE_URL)
    rewards = []
    steps_taken = 0
    total_reward = 0.0
    success = False
    score = 0.0

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action_type = get_model_action(step, TASK_NAME, result.observation.signals, result.observation.completed_actions)
            signal_index = 0 if action_type != "synthesize" else None
            action = DevPulseAction(type=action_type, signal_index=signal_index)
            
            result = await env.step(action)
            reward = result.reward or 0.0
            done = result.done
            error = "null"  # Capture if needed
            
            rewards.append(reward)
            total_reward += reward
            steps_taken = step
            
            log_step(step=step, action=action_type, reward=reward, done=done, error=error)
            
            if done:
                break
        
        # Normalize score to 0.0-1.0
        score = min(max(total_reward / 2.0, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        await env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())