from datasets import concatenate_datasets, load_dataset


def import_openbookqa():
    return load_dataset("allenai/openbookqa", "main")


def prepare_openbookqa():
    dataset = import_openbookqa()
    combined = concatenate_datasets(list(dataset.values()))

    def format_question(example):
        choices = example["choices"]["text"]
        prompt = "\n".join(
            [
                example["question_stem"],
                f"A) {choices[0]}",
                f"B) {choices[1]}",
                f"C) {choices[2]}",
                f"D) {choices[3]}",
                "Work through the solution, then state the final correct choice (A, B, C, or D).",
            ]
        )
        return {"id": example["id"], "prompt": prompt, "answerKey": example["answerKey"]}

    processed = combined.map(format_question, remove_columns=combined.column_names)
    print("dataset length:")
    print(len(processed))
    return processed
