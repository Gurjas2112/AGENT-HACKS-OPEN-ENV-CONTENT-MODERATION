import uuid
from typing import List, Dict, Any
from openenv.core.env_server import Environment
from ..models import DevPulseAction, DevPulseObservation, DevPulseState

class DevPulseCodeReviewEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, task: str = "easy"):
        self.task = task
        self.signals = []
        self.current_step = 0
        self.completed_actions = []
        self.done = False
        self.max_steps = {"easy": 5, "medium": 10, "hard": 15}[task]
        self.state_data = {}
        self.total_reward = 0.0
        self._load_signals()

    def _load_signals(self):
        """Load mock code signals (GitHub PRs/issues)."""
        self.signals = [
            {"id": "pr-001", "source": "github", "title": "Fix bug in auth module", "description": "Patch for CVE-2024-1234", "metadata": {"stars": 100}},
            {"id": "pr-002", "source": "github", "title": "Add new feature", "description": "Breaking change in API", "metadata": {"stars": 50}},
            {"id": "pr-003", "source": "github", "title": "Update documentation", "description": "Minor changes", "metadata": {"stars": 20}},
            {"id": "pr-004", "source": "github", "title": "Security fix", "description": "Critical vulnerability patch", "metadata": {"stars": 200}},
            {"id": "pr-005", "source": "github", "title": "Refactor code", "description": "No breaking changes", "metadata": {"stars": 30}},
        ]

    def reset(self, seed=None, episode_id=None, **kwargs) -> DevPulseObservation:
        self.current_step = 0
        self.completed_actions = []
        self.done = False
        self.state_data = {}
        self.total_reward = 0.0
        return DevPulseObservation(
            done=False,
            reward=0.0,
            task=self.task,
            signals=self.signals,
            current_step=self.current_step,
            completed_actions=self.completed_actions,
        )

    def step(self, action: DevPulseAction, timeout_s=None, **kwargs):
        if self.done or self.current_step >= self.max_steps:
            return DevPulseObservation(
                done=True,
                reward=0.0,
                task=self.task,
                signals=self.signals,
                current_step=self.current_step,
                completed_actions=self.completed_actions,
            )

        reward = 0.0
        self.completed_actions.append(action.type)
        self.current_step += 1

        if action.type == "score_relevance" and action.signal_index is not None and action.signal_index < len(self.signals):
            signal = self.signals[action.signal_index]
            score = 70 if "bug" in signal["title"].lower() or "fix" in signal["title"].lower() else 50
            self.state_data[f"relevance_{action.signal_index}"] = {"score": score}
            reward = 0.2 if score > 50 else 0.0
        elif action.type == "assess_risk" and action.signal_index is not None and action.signal_index < len(self.signals):
            signal = self.signals[action.signal_index]
            risk_level = "HIGH" if "security" in signal["title"].lower() or "cve" in signal["description"].lower() else "LOW"
            self.state_data[f"risk_{action.signal_index}"] = {"risk_level": risk_level}
            reward = 0.3 if risk_level == "HIGH" else 0.0
        elif action.type == "synthesize":
            assessed_signals = len([k for k in self.state_data if "relevance" in k or "risk" in k])
            reward = min(assessed_signals / 5.0, 1.0)  # Reward based on assessments done
            self.done = True
        else:
            reward = -0.1  # Penalty for invalid actions

        self.total_reward += reward
        if self.current_step >= self.max_steps:
            self.done = True

        return DevPulseObservation(
            done=self.done,
            reward=reward,
            task=self.task,
            signals=self.signals,
            current_step=self.current_step,
            completed_actions=self.completed_actions,
        )

    @property
    def state(self) -> DevPulseState:
        return DevPulseState(
            episode_id=str(uuid.uuid4()),
            step_count=self.current_step,
            task=self.task,
            completed_actions=self.completed_actions,
            signals_processed=len(self.state_data),
            total_reward=self.total_reward,
        )

    # Graders return 0.0-1.0 scores
    def grade_easy(self, actions: List[str]) -> float:
        relevance_count = sum(1 for a in actions if a == "score_relevance")
        return min(relevance_count / 3.0, 1.0)  # At least 3 relevance scores

    def grade_medium(self, actions: List[str]) -> float:
        risk_count = sum(1 for a in actions if a == "assess_risk")
        loop_penalty = -0.1 * sum(1 for i in range(1, len(actions)) if actions[i] == actions[i-1])  # Penalize repeats
        return max(0.0, min(risk_count / 5.0 + loop_penalty, 1.0))  # At least 5 risk assessments, penalize loops

    def grade_hard(self, actions: List[str]) -> float:
        has_synthesis = "synthesize" in actions
        prior_actions = len([a for a in actions if a != "synthesize"])
        return 1.0 if has_synthesis and prior_actions >= 10 else max(0.0, prior_actions / 10.0)  # Synthesis after 10+ actions