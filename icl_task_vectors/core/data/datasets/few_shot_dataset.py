from typing import List


class FewShotDataset:
    def __init__(self,
        train_inputs: List[str],
        train_outputs: List[str],
        valid_inputs: List[str],
        valid_outputs: List[str],
        test_input: str,
        test_output: str
    ):
        self.train_inputs = train_inputs
        self.train_outputs = train_outputs
        self.valid_inputs = valid_inputs
        self.valid_outputs = valid_outputs
        self.test_input = test_input
        self.test_output = test_output
