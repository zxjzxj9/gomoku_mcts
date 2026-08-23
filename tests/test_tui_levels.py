import json

import numpy as np
import pytest

from gomoku.difficulty import load_levels
from gomoku.tui.app import GomokuApp


def app_with_levels(tmp_path, **kwargs):
    path = tmp_path / "elo.json"
    path.write_text(json.dumps({"ratings": {
        "level1": 800, "level2": 1050, "level3": 1300,
        "level4": 1550, "level5": 1800}}))
    defaults = dict(size=5, win_length=5, levels=load_levels(path),
                    rng=np.random.default_rng(0))
    defaults.update(kwargs)
    return GomokuApp(**defaults)


async def test_header_shows_the_level_name_and_elo(tmp_path):
    app = app_with_levels(tmp_path, level_index=3)
    async with app.run_test():
        text = str(app.query_one("#level").content)
        assert "Club" in text and "1300" in text


async def test_number_keys_switch_level(tmp_path):
    app = app_with_levels(tmp_path, level_index=3)
    async with app.run_test() as pilot:
        await pilot.press("5")
        assert app.level.index == 5
        assert "Expert" in str(app.query_one("#level").content)


async def test_unrated_levels_are_labelled_rather_than_invented():
    app = GomokuApp(size=5, win_length=5, levels=load_levels(None),
                    rng=np.random.default_rng(0))
    async with app.run_test():
        assert "unrated" in str(app.query_one("#level").content)


async def test_pc_vs_pc_mode_plays_itself_without_human_input(tmp_path):
    app = app_with_levels(tmp_path, mode="pc-vs-pc", level_index=1)
    async with app.run_test() as pilot:
        await pilot.pause()
        for _ in range(6):
            await app.workers.wait_for_complete()
            await pilot.pause()
        assert app.state.ply >= 2


async def test_changing_level_starts_a_new_game(tmp_path):
    app = app_with_levels(tmp_path, level_index=2)
    async with app.run_test() as pilot:
        await pilot.press("enter")
        await pilot.press("4")
        assert app.state.ply == 0
        assert app.level.index == 4
