"""Test AgentOptimizer candidate-pool behavior."""

import sys
sys.path.insert(0, '../src')

from icl_agent.core.optimizer import AgentOptimizer


class StaticImprovementStrategy:
    def learn(self, current_config, contexts, failed_contexts, knowledge_extractor):
        new_config = current_config.copy()
        new_config["system_prompt"] = "improved prompt"
        return new_config


class PromptAwareAdapter:
    def execute(self, agent_config, data):
        prompt = agent_config.get("system_prompt", "")
        if "improved" in prompt:
            return {"answer": data["answer"]}
        return {"answer": "wrong"}


def exact_match_evaluator(output, data):
    return 1.0 if output.get("answer") == data.get("answer") else 0.0


def test_optimizer_tracks_prompt_candidates():
    optimizer = AgentOptimizer(
        initial_agent_config={"system_prompt": "base prompt"},
        learning_strategy=StaticImprovementStrategy(),
        adapter=PromptAwareAdapter(),
        evaluator=exact_match_evaluator,
        max_iterations=1,
        failure_threshold=1.0,
        verbose=False,
        candidate_pool_size=4,
    )

    result = optimizer.optimize(
        trainset=[{"question": "q1", "answer": "a1"}],
        valset=[{"question": "q1", "answer": "a1"}],
    )

    prompts = [candidate.system_prompt for candidate in result.candidate_pool]
    assert "base prompt" in prompts
    assert "improved prompt" in prompts
    assert result.best_instruction == "improved prompt"
    assert result.metadata["candidate_pool"]["num_candidates"] == 2
    assert result.iteration_logs[0]["candidate_pool_size"] == 2
    assert result.to_dict()["candidate_pool"]
    print("PASS: optimizer_tracks_prompt_candidates")


if __name__ == "__main__":
    print("Running Optimizer tests...\n")
    test_optimizer_tracks_prompt_candidates()
    print("\nAll Optimizer tests passed!")
