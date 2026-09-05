"""LM (planner) adapter training for the ACE-Step 5Hz language model.


Trains LoRA adapters for the ``acestep-5Hz-lm-*`` planner so that song
structure / phrasing / vocal placement can be specialized per artist,
complementing the DiT adapter (which captures timbre/production).

Pipeline: extract (dataset songs -> code-sequence JSONL) -> train (PEFT
LoRA SFT) -> export (PEFT adapter and/or merged HF checkpoint dir into
the models root, where a compatible inference engine loads it directly).
"""
