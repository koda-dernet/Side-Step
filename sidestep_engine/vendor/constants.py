"""ACE-Step constants required by Side-Step (vendored subset)."""

DEFAULT_DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"

SFT_GEN_PROMPT = """# Instruction
{}

# Caption
{}

# Metas
{}<|endoftext|>
"""
