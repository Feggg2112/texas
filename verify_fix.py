# -*- coding: utf-8 -*-
"""
快速验证脚本 - 检查修复是否有效
"""

print("=" * 70)
print("  验证修复")
print("=" * 70)

# 测试 1：检查 GameState 定义
print("\n[测试 1] 检查 GameState 定义...")
try:
    from game_nodes import GameState
    print("  ✓ GameState 导入成功")
    print(f"  ✓ 字段：{list(GameState.__annotations__.keys())}")
except Exception as e:
    print(f"  ✗ 错误：{e}")

# 测试 2：检查节点函数
print("\n[测试 2] 检查节点函数...")
try:
    from game_nodes import dealer_node, agent_laoying_action, check_hand_end
    print("  ✓ 节点函数导入成功")
except Exception as e:
    print(f"  ✗ 错误：{e}")

# 测试 3：检查 Agent 提示词
print("\n[测试 3] 检查 Agent 提示词...")
try:
    from agent_config import AGENT_PROFILES
    for name, profile in AGENT_PROFILES.items():
        prompt = profile["system_prompt"]
        has_fold = "fold" in prompt or "弃牌" in prompt
        has_raise = "raise" in prompt or "加注" in prompt
        print(f"  ✓ {name}：弃牌={has_fold}，加注={has_raise}")
except Exception as e:
    print(f"  ✗ 错误：{e}")

# 测试 4：模拟一个简单的游戏流程
print("\n[测试 4] 模拟游戏流程...")
try:
    from game_nodes import GameState, dealer_node
    from poker_core import create_deck
    
    # 创建初始状态
    initial_state: GameState = {
        "hand_number": 1,
        "chips": {"老鹰": 1000, "小辣椒": 1000, "老钱": 1000},
        "pot": 0,
        "community_cards": [],
        "players_info": {},
        "messages": [],
        "current_player": "",
        "stage": "preflop",
        "hand_ended": False,
        "winner": "",
        "round_count": 0,
    }
    
    # 执行发牌节点
    result = dealer_node(initial_state)
    print(f"  ✓ 发牌成功")
    print(f"    - 底池：{result['pot']}")
    print(f"    - 玩家数：{len(result['players_info'])}")
    print(f"    - 轮数：{result['round_count']}")
    
    # 检查玩家信息
    for name, info in result['players_info'].items():
        print(f"    - {name}：{info['hole_cards'][0]} {info['hole_cards'][1]}")
    
except Exception as e:
    print(f"  ✗ 错误：{e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("  验证完成！")
print("  现在可以运行：python texas_poker_game.py")
print("=" * 70)
