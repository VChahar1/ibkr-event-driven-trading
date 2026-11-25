"""Plot actual pytest outcomes from the committed validation record."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    """Write a validation chart without running a broker or simulation."""
    root = Path(__file__).resolve().parents[1]
    record = json.loads((root / "results/validation.json").read_text())
    unit = record["offline"]
    groups = unit["passed_by_group"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    bars = axes[0].barh(list(groups), list(groups.values()), color="#176b60")
    axes[0].bar_label(bars, padding=4)
    axes[0].set(xlabel="Passed test cases", title="Offline validation by module group")
    maximum = max(groups.values(), default=1)
    axes[0].set_xlim(0, maximum * 1.2)
    integrations = record["integration"]
    values = [integrations["passed"], integrations["skipped"], integrations["failed"]]
    bars = axes[1].bar(["Passed", "Skipped", "Failed"], values, color=["#547da8", "#9aa6b2", "#a7513b"])
    axes[1].bar_label(bars, padding=4)
    axes[1].set(
        ylabel="Test cases",
        title="Real Kafka/Redis integration checks",
        ylim=(0, max(values, default=1) * 1.3 + 1),
    )
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    (root / "assets").mkdir(exist_ok=True)
    fig.savefig(root / "assets/test-results.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
