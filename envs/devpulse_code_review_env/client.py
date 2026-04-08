from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from .models import DevPulseAction, DevPulseObservation, DevPulseState

class DevPulseCodeReviewEnv(EnvClient[DevPulseAction, DevPulseObservation, DevPulseState]):
    def _step_payload(self, action: DevPulseAction) -> dict:
        return {"type": action.type, "signal_index": action.signal_index}

    def _parse_result(self, payload: dict) -> StepResult[DevPulseObservation]:
        obs_data = payload.get("observation", {})
        obs = DevPulseObservation(
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            task=obs_data.get("task", ""),
            signals=obs_data.get("signals", []),
            current_step=obs_data.get("current_step", 0),
            completed_actions=obs_data.get("completed_actions", []),
        )
        return StepResult(
            observation=obs,
            reward=obs.reward,
            done=obs.done,
        )

    def _parse_state(self, payload: dict) -> DevPulseState:
        return DevPulseState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task=payload.get("task", ""),
            completed_actions=payload.get("completed_actions", []),
            signals_processed=payload.get("signals_processed", 0),
            total_reward=payload.get("total_reward", 0.0),
        )

    # Add sync wrappers for async
    def reset(self):
        return asyncio.run(super().reset())

    def step(self, action):
        return asyncio.run(super().step(action))