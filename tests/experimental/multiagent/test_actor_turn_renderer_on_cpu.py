# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Validates ``ActorTurnRenderer``'s core correctness invariant: the token sequence it
incrementally accumulates for one trainable actor across a multi-agent episode must be
byte-identical to what a real generation call was actually conditioned on. If these diverge,
``old_log_probs`` recomputed from the stored sequence at training time no longer corresponds to
what the rollout policy actually sampled, silently corrupting the PPO importance ratio.
"""

import unittest

from transformers import AutoTokenizer

from verl.experimental.multiagent.agent_loop import ActorTurnRenderer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class TestActorTurnRendererEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL)

    def _generated(self, text: str) -> list[int]:
        # Real generation includes the model's own stop token in its output (it's the model's
        # stop token) but never the chat template's trailing turn separator after it -- see
        # initialize_turn_separator's docstring in verl/utils/tokenizer/chat_template.py.
        return self.tokenizer.encode(text, add_special_tokens=False) + [self.tokenizer.eos_token_id]

    def test_own_generation_prompt_matches_a_fresh_generation_prompt_call(self):
        renderer = ActorTurnRenderer(self.tokenizer)
        messages = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hi"}]

        prompt_ids = renderer.build_initial_prompt(messages)
        incremental_generation_prompt = list(prompt_ids) + renderer.generation_marker

        fresh_generation_prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        assert incremental_generation_prompt == fresh_generation_prompt

    def test_observation_turn_matches_full_retemplate_at_that_point(self):
        renderer = ActorTurnRenderer(self.tokenizer)
        system = {"role": "system", "content": "You are helpful."}
        user = {"role": "user", "content": "What is 2+2?"}
        own_text = "The answer is 4."
        observation = {"role": "user", "content": "[oracle] Good job."}

        buf = list(renderer.build_initial_prompt([system, user]))
        buf += renderer.generation_marker + self._generated(own_text)
        buf += renderer.render_observation(observation)

        full_history = [system, user, {"role": "assistant", "content": own_text}, observation]
        expected = self.tokenizer.apply_chat_template(full_history, add_generation_prompt=False, tokenize=True)
        assert buf == expected

    def test_three_turn_chain_matches_what_generation_actually_saw(self):
        """Own turn -> observation -> own turn again: the buffer right before the third turn's
        generation must equal a real add_generation_prompt=True call over the same history."""
        renderer = ActorTurnRenderer(self.tokenizer)
        system = {"role": "system", "content": "You are helpful."}
        user = {"role": "user", "content": "What is 2+2?"}
        first_text = "The answer is 4."
        observation = {"role": "user", "content": "[oracle] Good job."}
        second_text = "Thanks!"

        buf = list(renderer.build_initial_prompt([system, user]))
        buf += renderer.generation_marker + self._generated(first_text)
        buf += renderer.render_observation(observation)
        buf += renderer.generation_marker + self._generated(second_text)

        history_before_second_turn = [system, user, {"role": "assistant", "content": first_text}, observation]
        real_prompt_for_second_turn = self.tokenizer.apply_chat_template(
            history_before_second_turn, add_generation_prompt=True, tokenize=True
        )
        expected = list(real_prompt_for_second_turn) + self._generated(second_text)
        assert buf == expected


if __name__ == "__main__":
    unittest.main()
