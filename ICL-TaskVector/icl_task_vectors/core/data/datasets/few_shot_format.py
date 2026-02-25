from core.data.datasets.few_shot_dataset import FewShotDataset
from typing import Optional, List


class FewShotFormat:
    def __init__(
        self,
        example_format: str = "example:{input}->{output}",
        task_format: str = "example:{input}{arrow}{output}",
        # example_format: str = "input:{input}, output:{output}",
        example_separator: str = "\n",
        task_description: Optional[str] = None,
        test_example_format: Optional[str] = "example:{input}->",
        # test_example_format: Optional[str] = "input:{input}, output:",
    ):
        self.example_format = example_format
        self.example_separator = example_separator
        self.task_description = task_description
        self.test_example_format = test_example_format
        self.task_format = task_format

    def format_train_example(self, inp: str, out: str) -> str:
        return self.example_format.format(input=inp, output=out)

    def format_test_example(self, inp: str) -> str:
        if self.test_example_format is None:
            return self.format_train_example(inp, "")
        else:
            return self.test_example_format.format(input=inp)

    def format_task_example(self, inp: str, out: str, num_dup: int) -> str:
        return self.task_format.format(input=inp, arrow='->' * num_dup, output=out)

    def format_datasets(self, datasets: List[FewShotDataset], **kwargs) -> List[str]:
        return [self.format_dataset(dataset, **kwargs) for dataset in datasets]

    def format_dataset(self, dataset: FewShotDataset, include_train: bool = True, include_valid: bool = False,
        include_test: bool = True, num_dup: int = 1) -> str:

        base_prompt = ""
        if self.task_description is not None:
            base_prompt += f"{self.task_description}{self.example_separator}"

        prompt = base_prompt
        if include_train:
            if len(dataset.train_inputs) > 0:
                train_examples = [
                    self.format_train_example(x, y) for x, y in zip(dataset.train_inputs, dataset.train_outputs)
                ]
                train_examples_prompt = self.example_separator.join(train_examples)
                train_examples_prompt += self.example_separator
            else:
                train_examples_prompt = ""

            prompt += train_examples_prompt

        if include_valid:
            if len(dataset.valid_inputs) > 0:
                valid_examples_prompt = self.format_task_example(dataset.valid_inputs[0], dataset.valid_outputs[0], num_dup=num_dup)
                valid_examples_prompt += self.example_separator

                if len(dataset.valid_inputs) > 1:
                    valid_examples = [
                        self.format_train_example(x, y) for x, y in zip(dataset.valid_inputs[1:], dataset.valid_outputs[1:])
                    ]
                    valid_examples_prompt += self.example_separator.join(valid_examples)
                    valid_examples_prompt += self.example_separator
            else:
                valid_examples_prompt = ""

            prompt += valid_examples_prompt

        if include_test:
            test_example_prompt = self.format_test_example(dataset.test_input)
            prompt += test_example_prompt

        return prompt
