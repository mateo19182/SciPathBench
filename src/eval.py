# evaluation.py
# Runs the agent against the benchmark and calculates performance metrics.

import logging


class EvaluationHarness:
    """Calculates performance metrics for a single benchmark run."""

    def __init__(self, ground_truth_path: list, agent_path: list, agent_steps: int):
        self.ground_truth_path = ground_truth_path
        self.agent_path = agent_path
        self.agent_steps = agent_steps
        self.scorecard = {}

    def run_evaluation(self):
        """Calculates all metrics and returns the scorecard."""
        logging.info("--- Running Evaluation ---")
        self._calculate_path_success()
        self._calculate_path_optimality()
        self._calculate_reasoning_faithfulness()
        return self.scorecard

    def _calculate_path_success(self):
        self.scorecard["path_success"] = 1 if self.agent_path else 0
        logging.info(f"Path Success: {self.scorecard['path_success']}")

    def _calculate_path_optimality(self):
        if not self.scorecard["path_success"] or not self.ground_truth_path:
            self.scorecard["path_optimality"] = 0
        else:
            len_agent = len(self.agent_path) - 1
            len_true = len(self.ground_truth_path) - 1
            self.scorecard["path_optimality"] = (
                len_true / len_agent if len_agent > 0 else 1.0
            )
        logging.info(f"Path Optimality: {self.scorecard['path_optimality']:.2f}")

    def _calculate_reasoning_faithfulness(self):
        # TODO: Implement a function to verify each link in the agent's path
        # by making API calls. For now, we assume it's faithful if a path was found.
        self.scorecard["reasoning_faithfulness"] = 1 if self.agent_path else 0
        logging.info(
            f"Reasoning Faithfulness: {self.scorecard['reasoning_faithfulness']}"
        )
