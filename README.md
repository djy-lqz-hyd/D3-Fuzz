# D-Fuzz
D3Fuzz first executes an official test suite to collect coverage and identify low-coverage files and critical lines, then assigns weights to these files and employs a dynamic priority scoring function to calculate their coverage score, finally prompts LLM to generate high-quality test cases to explore unknown input spaces.
