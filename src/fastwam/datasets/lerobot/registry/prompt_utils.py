import os

TEXT_EMBEDS_SUBDIR = "text_embeds"


def build_prompt(task: str, embodiment: str, views: str, control: str) -> str:
    return f"{embodiment}, {views}, {control}. {task}"


def build_prompt_from_config(task: str, data_config) -> str:
    return build_prompt(
        task,
        data_config.prompt_embodiment,
        data_config.prompt_views,
        data_config.prompt_control,
    )


def get_text_embeds_dir(dataset_dir: str) -> str:
    return os.path.join(dataset_dir, TEXT_EMBEDS_SUBDIR)
