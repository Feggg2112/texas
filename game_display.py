# -*- coding: utf-8 -*-
"""
游戏显示和工具函数
"""


def print_game_state(state):
    """打印游戏状态"""
    print("\n" + "=" * 70)
    print(f"【第 {state['hand_number']} 手】")
    print(f"底池：{state['pot']} 筹码")
    chips_str = " | ".join([f"{n} {state['chips'][n]}" for n in ["老鹰", "小辣椒", "老钱"]])
    print(f"筹码：{chips_str}")
    print("=" * 70)


def print_player_action(agent_name: str, action: str, amount: int, speech: str):
    """打印玩家行动"""
    action_cn = {
        "fold": "弃牌",
        "check": "过牌",
        "call": "跟注",
        "raise": "加注"
    }.get(action, action)
    
    print(f"\n🎴 {agent_name}")
    if action == "raise":
        print(f"   行动：{action_cn} {amount} 筹码")
    else:
        print(f"   行动：{action_cn}")
    
    if speech:
        print(f"   说话：{speech}")


def print_dealer_info(players_info: dict):
    """打印发牌信息"""
    print("\n" + "▓" * 70)
    print("▓ 发牌中...")
    print("▓" * 70)
    
    for name in ["老鹰", "小辣椒", "老钱"]:
        if name in players_info:
            hole_cards = players_info[name]["hole_cards"]
            print(f"  {name} 的手牌：{hole_cards[0]} {hole_cards[1]}")


def print_winner_info(winner: str, pot: int, chips: dict):
    """打印赢家信息"""
    print("\n" + "▓" * 70)
    print("▓ 结算中...")
    print("▓" * 70)
    
    print(f"\n🏆 赢家：{winner}")
    print(f"   赢得底池：{pot} 筹码")
    print(f"\n筹码更新：")
    for name in ["老鹰", "小辣椒", "老钱"]:
        print(f"  {name}：{chips[name]} 筹码")
