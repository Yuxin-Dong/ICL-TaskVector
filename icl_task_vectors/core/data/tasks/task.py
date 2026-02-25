import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Iterable

from core.data.datasets.few_shot_dataset import FewShotDataset
from transformers import PreTrainedTokenizer


class Task(ABC):
    def __init__(self, tokenizer: PreTrainedTokenizer, allow_prefix: bool = False):
        self.tokenizer = tokenizer
        self.allow_prefix = allow_prefix

    @abstractmethod
    def sample_inputs(self, num_inputs: int, exclude: Optional[Iterable[Any]] = ()) -> List[Any]:
        pass

    @abstractmethod
    def calc_output(self, inp: Any) -> Any:
        pass

    @abstractmethod
    def num_examples(self) -> int:
        pass

    def compare_outputs(self, output1: Any, output2: Any) -> bool:
        output1, output2 = output1.strip(), output2.strip()

        if self.allow_prefix:
            nonempy = len(output1) > 0 and len(output2) > 0
            return nonempy and (output1.startswith(output2) or output2.startswith(output1))
        return output1 == output2

    def calc_test_output(self, inp: Any) -> Any:
        return self.calc_output(inp)

    def create_datasets(self, num_datasets: int, num_train: int, num_valid: int, same_test: Optional[List[FewShotDataset]] = None) -> List[FewShotDataset]:
        if same_test is None:
            return [self.create_dataset(num_train, num_valid) for _ in range(num_datasets)]
        else:
            return [self.create_dataset(num_train, num_valid, same_test[_].test_input) for _ in range(num_datasets)]

    def create_dataset(self, num_train: int, num_valid: int, test_input: Optional[Any] = None) -> FewShotDataset:
        if test_input is None:
            test_input = self.sample_inputs(1)[0]
        test_output = self.calc_test_output(test_input)

        train_inputs = self.sample_inputs(num_train, exclude=[test_input])
        train_outputs = [self.calc_output(x) for x in train_inputs]

        valid_inputs = self.sample_inputs(num_valid, exclude=train_inputs+[test_input])
        valid_outputs = [self.calc_output(x) for x in valid_inputs]

        train_inputs = [str(x) for x in train_inputs]
        train_outputs = [str(x) for x in train_outputs]
        valid_inputs = [str(x) for x in valid_inputs]
        valid_outputs = [str(x) for x in valid_outputs]
        test_input = str(test_input)
        test_output = str(test_output)

        return FewShotDataset(
            train_inputs,
            train_outputs,
            valid_inputs,
            valid_outputs,
            test_input,
            test_output,
        )
