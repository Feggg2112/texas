# -*- coding: utf-8 -*-
"""
快速测试脚本 - 验证模块导入和基础功能
"""

import sys

print("=" * 70)
print("  德州扑克多Agent游戏 - 模块测试")
print("=" * 70)

# 测试 1：导入 poker_core
print("\n[测试 1] 导入 poker_core 模块...")
try:
    from poker_core import Card, create_deck, evaluate_hand, determine_winner
    print("  ✓ poker_core 导入成功")
    
    # 测试创建牌
    card = Card("♠", "A")
    print(f"  ✓ 创建牌成功：{card}")
    
    # 测试创建牌组
    deck = create_deck()
    print(f"  ✓ 创建牌组成功，共 {len(deck)} 张牌")
except Exception as e:
    print(f"  ✗ 错误：{e}")
    sys.exit(1)

# 测试 2：导入 agent_config
print("\n[测试 2] 导入 agent_config 模块...")
try:
    from agent_config import AGENT_PROFILES
    print("  ✓ agent_config 导入成功")
    print(f"  ✓ 加载了 {len(AGENT_PROFILES)} 个 Agent：{list(AGENT_PROFILES.keys())}")
except Exception as e:
    print(f"  ✗ 错误：{e}")
    sys.exit(1)

# 测试 3：导入 game_display
print("\n[测试 3] 导入 game_display 模块...")
try:
    from game_display import print_game_state, print_player_action
    print("  ✓ game_display 导入成功")
except Exception as e:
    print(f"  ✗ 错误：{e}")
    sys.exit(1)

# 测试 4：导入 game_nodes
print("\n[测试 4] 导入 game_nodes 模块...")
try:
    from game_nodes import GameState, dealer_node, check_hand_end
    print("  ✓ game_nodes 导入成功")
except Exception as e:
    print(f"  ✗ 错误：{e}")
    sys.exit(1)

# 测试 5：导入主程序
print("\n[测试 5] 导入 texas_poker_game 模块...")
try:
    from texas_poker_game import build_game_graph
    print("  ✓ texas_poker_game 导入成功")
except Exception as e:
    print(f"  ✗ 错误：{e}")
    sys.exit(1)

# 测试 6：测试手牌评估
print("\n[测试 6] 测试手牌评估...")
try:
    hole_cards = [Card("♠", "A"), Card("♥", "K")]
    community_cards = [Card("♦", "A"), Card("♣", "Q"), Card("♠", "J")]
    score, desc = evaluate_hand(hole_cards, community_cards)
    print(f"  ✓ 手牌评估成功")
    print(f"    手牌：{hole_cards[0]} {hole_cards[1]}")
    print(f"    公共牌：{community_cards[0]} {community_cards[1]} {community_cards[2]}")
    print(f"    牌型：{desc}（分数：{score}）")
except Exception as e:
    print(f"  ✗ 错误：{e}")
    sys.exit(1)

# 测试 7：测试赢家判断
print("\n[测试 7] 测试赢家判断...")
try:
    players_info = {
        "老鹰": {
            "hole_cards": [Card("♠", "A"), Card("♥", "A")],
            "folded": False,
        },
        "小辣椒": {
            "hole_cards": [Card("♦", "K"), Card("♣", "K")],
            "folded": False,
        },
        "老钱": {
            "hole_cards": [Card("♠", "Q"), Card("♥", "Q")],
            "folded": False,
        },
    }
    community_cards = [Card("♦", "2"), Card("♣", "3"), Card("♠", "4")]
    winner = determine_winner(players_info, community_cards)
    print(f"  ✓ 赢家判断成功：{winner}")
except Exception as e:
    print(f"  ✗ 错误：{e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("  所有测试通过！✓")
print("  现在可以运行：python texas_poker_game.py")
print("=" * 70)
