from __future__ import annotations

from pathlib import Path

from lisai.evaluation.run_evaluate import (
    _build_evaluation_folder_name,
    _expand_checkpoint_selection,
    _format_eval_gt_for_display,
)


def test_build_evaluation_folder_name_uses_requested_epoch_when_explicit():
    assert _build_evaluation_folder_name(
        best_or_last='best',
        requested_epoch=12,
        resolved_epoch=7,
        split='test',
    ) == 'evaluation_epoch_12'


def test_build_evaluation_folder_name_includes_selector_and_resolved_epoch():
    assert _build_evaluation_folder_name(
        best_or_last='last',
        requested_epoch=None,
        resolved_epoch=100,
        split='val',
    ) == 'evaluation_last_epoch_100_val'


def test_build_evaluation_folder_name_falls_back_when_epoch_is_unknown():
    assert _build_evaluation_folder_name(
        best_or_last='best',
        requested_epoch=None,
        resolved_epoch=None,
        split='test',
    ) == 'evaluation_best'


def test_expand_checkpoint_selection_for_both_creates_two_runs():
    options = {
        "best_or_last": "both",
        "epoch_number": None,
        "save_folder": Path("/tmp/eval"),
        "results": {"seed": 1},
    }

    expanded = _expand_checkpoint_selection(options)

    assert [item["best_or_last"] for item in expanded] == ["best", "last"]
    assert expanded[0]["save_folder"] == Path("/tmp/eval/best")
    assert expanded[1]["save_folder"] == Path("/tmp/eval/last")
    assert expanded[0]["results"] == {"seed": 1}
    assert expanded[1]["results"] == {"seed": 1}
    assert expanded[0]["results"] is not options["results"]
    assert expanded[1]["results"] is not options["results"]


def test_format_eval_gt_for_display_handles_special_values():
    assert _format_eval_gt_for_display(None) == "<none>"
    assert _format_eval_gt_for_display("") == "<root>"
    assert _format_eval_gt_for_display("gt_avg") == "gt_avg"
