# -*- coding: utf-8 -*-
"""
游戏显示和工具函数
"""


def _chips_display(name: str, chips: dict) -> str:
    """筹码显示，0 筹码显示为出局"""
    amount = chips.get(name, 0)
    if amount <= 0:
        return name + " 【出局】"
    return name + " " + str(amount)


def print_game_state(state):
    """打印游戏状态"""
    print("\n" + "=" * 70)
    print("【第 " + str(state["hand_number"]) + " 手】")
    print("底池：" + str(state["pot"]) + " 筹码")
    order = state.get("player_order", ["老鹰", "小辣椒", "老钱", "火线", "铁拳"])
    chips_str = " | ".join([_chips_display(n, state["chips"]) for n in order])
    print("筹码：" + chips_str)
    print("=" * 70)


def print_player_action(agent_name: str, action: str, amount: int, speech: str, chips_after_action: int, hole_cards: list, show_hole_cards: bool = True):
    """打印玩家行动"""
    action_cn = {
        "fold": "弃牌",
        "check": "过牌",
        "call": "跟注",
        "raise": "加注"
    }.get(action, action)

    cards_text = ""
    if show_hole_cards and hole_cards and len(hole_cards) >= 2:
        cards_text = "（" + str(hole_cards[0]) + " " + str(hole_cards[1]) + "）"
    print("【" + agent_name + cards_text + "】")
    if action in ("raise", "call"):
        print("   行动：" + action_cn + " " + str(amount) + " 筹码" + "（剩余筹码：" + str(chips_after_action) + "）")
    else:
        print("   行动：" + action_cn)
    if speech:
        print("   说话：" + speech)


def print_dealer_info(players_info: dict, mode: str = "god", human_player: str = "你"):
    """打印发牌信息"""
    print("\n" + "▓" * 70)
    print("▓ 发牌中...")
    print("▓" * 70)
    for name, info in players_info.items():
        hole_cards = info["hole_cards"]
        if info.get("folded", False) and len(hole_cards) < 2:
            print("  " + name + "：已出局，本手不参与")
        else:
            can_show = (mode == "god") or (name == human_player)
            if can_show:
                print("  " + name + " 的手牌：" + str(hole_cards[0]) + " " + str(hole_cards[1]))
            else:
                print("  " + name + " 的手牌：** **")


def print_showdown_info(players_info: dict, community_cards: list):
    """摊牌时展示所有未弃牌玩家手牌"""
    print("\n" + "─" * 70)
    print("【摊牌】公共牌：" + " ".join([str(c) for c in community_cards]))
    for name, info in players_info.items():
        if info.get("folded", False):
            continue
        hole_cards = info.get("hole_cards", [])
        if len(hole_cards) >= 2:
            print("  " + name + "：" + str(hole_cards[0]) + " " + str(hole_cards[1]))
    print("─" * 70)


def print_community_cards(stage: str, community_cards: list):
    """打印公共牌"""
    stage_cn = {
        "flop": "翻牌",
        "turn": "转牌",
        "river": "河牌",
    }.get(stage, stage)
    cards_str = " ".join([str(c) for c in community_cards])
    print("\n" + "─" * 70)
    print("【" + stage_cn + "】公共牌：" + cards_str)
    print("─" * 70)


def print_winner_info(winner: str, pot: int, chips: dict):
    """打印赢家信息"""
    print("\n" + "▓" * 70)
    print("▓ 结算中...")
    print("▓" * 70)
    print("\n🏆 赢家：" + winner)
    print("   赢得底池：" + str(pot) + " 筹码")
    print("\n筹码更新：")
    for name, amount in chips.items():
        if amount <= 0:
            print("  " + name + "：0 筹码 【出局】")
        else:
            print("  " + name + "：" + str(amount) + " 筹码")
