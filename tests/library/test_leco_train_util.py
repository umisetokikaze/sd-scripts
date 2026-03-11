from pathlib import Path

from library.leco_train_util import load_prompt_settings


def test_load_prompt_settings_with_original_format(tmp_path: Path):
    prompt_file = tmp_path / "prompts.yaml"
    prompt_file.write_text(
        """
- target: "van gogh"
  guidance_scale: 1.5
  resolution: 512
""".strip(),
        encoding="utf-8",
    )

    prompts = load_prompt_settings(prompt_file)

    assert len(prompts) == 1
    assert prompts[0].target == "van gogh"
    assert prompts[0].positive == "van gogh"
    assert prompts[0].unconditional == ""
    assert prompts[0].neutral == ""
    assert prompts[0].action == "erase"
    assert prompts[0].guidance_scale == 1.5


def test_load_prompt_settings_with_slider_targets(tmp_path: Path):
    prompt_file = tmp_path / "slider.yaml"
    prompt_file.write_text(
        """
targets:
  - target_class: ""
    positive: "high detail"
    negative: "low detail"
    multiplier: 1.25
    weight: 0.5
guidance_scale: 2.0
resolution: 768
neutral: ""
""".strip(),
        encoding="utf-8",
    )

    prompts = load_prompt_settings(prompt_file)

    assert len(prompts) == 4

    first = prompts[0]
    second = prompts[1]
    third = prompts[2]
    fourth = prompts[3]

    assert first.target == ""
    assert first.positive == "low detail"
    assert first.unconditional == "high detail"
    assert first.action == "erase"
    assert first.multiplier == 1.25
    assert first.weight == 0.5
    assert first.get_resolution() == (768, 768)

    assert second.positive == "high detail"
    assert second.unconditional == "low detail"
    assert second.action == "enhance"
    assert second.multiplier == 1.25

    assert third.action == "erase"
    assert third.multiplier == -1.25

    assert fourth.action == "enhance"
    assert fourth.multiplier == -1.25
