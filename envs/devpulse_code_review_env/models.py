from typing import List, Dict, Any, Optional
from openenv.core.env_server import Action, Observation, State

class DevPulseAction(Action):
    type: str  # "score_relevance", "assess_risk", "synthesize"
    signal_index: Optional[int] = None  # For individual signal actions

class DevPulseObservation(Observation):
    # done: bool and reward: Optional[float] inherited
    task: str
    signals: List[Dict[str, Any]]
    current_step: int
    completed_actions: List[str]

class DevPulseState(State):
    # episode_id: Optional[str] and step_count: int inherited
    task: str
    completed_actions: List[str]
    signals_processed: int
    total_reward: float