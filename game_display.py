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
    order = state.get("player_order", ["老鹰", "小辣椒", "老钱", "火线" , "铁拳"])
    chips_str = " | ".join([_chips_display(n, state["chips"]) for n in order])
    print("筹码：" + chips_str)
    print("=" * 70)


def print_player_action(agent_name: str, action: str, amount: int, speech: str, chips_after_action: int, hole_cards: list):
    """打印玩家行动"""
    action_cn = {
        "fold": "弃牌",
        "check": "过牌",
        "call": "跟注",
        "raise": "加注"
    }.get(action, action)

    cards_text = ""
    if hole_cards and len(hole_cards) >= 2:
        cards_text = "（" + str(hole_cards[0]) + " " + str(hole_cards[1]) + "）"
    print("\u3014" + agent_name + cards_text + "\u3015")
    if action in ("raise", "call"):
        print("   行动：" + action_cn + " " + str(amount) + " 筹码" + "（剩余筹码：" + str(chips_after_action) + "）")
    else:
        print("   行动：" + action_cn)
    if speech:
        print("   说话：" + speech)


def print_dealer_info(players_info: dict):
    """打印发牌信息"""
    print("\n" + "\u2593" * 70)
    print("\u2593 发牌中...")
    print("\u2593" * 70)
    for name, info in players_info.items():
        hole_cards = info["hole_cards"]
        if info.get("folded", False) and len(hole_cards) < 2:
            print("  " + name + "：已出局，本手不参与")
        else:
            print("  " + name + " 的手牌：" + str(hole_cards[0]) + " " + str(hole_cards[1]))


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
    print("\n" + "\u2593" * 70)
    print("\u2593 结算中...")
    print("\u2593" * 70)
    print("\n\U0001f3c6 赢家：" + winner)
    print("   赢得底池：" + str(pot) + " 筹码")
    print("\n筹码更新：")
    for name, amount in chips.items():
        if amount <= 0:
            print("  " + name + "：0 筹码 【出局】")
        else:
            print("  " + name + "：" + str(amount) + " 筹码")
