from pathlib import Path

from runtime.skills import SkillCatalog


def test_skill_catalog_discovers_bounded_skill_files(tmp_path: Path) -> None:
    skill = tmp_path / "skills" / "review"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: secure-review\ndescription: Review code safely\n---\nCheck tests first.\n",
        encoding="utf-8",
    )
    catalog = SkillCatalog([tmp_path / "skills"])
    assert catalog.list()[0]["name"] == "secure-review"
    assert "Check tests first" in catalog.prompt_context(["secure-review"])
