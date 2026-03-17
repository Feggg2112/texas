# -*- coding: utf-8 -*-
"""
德州扑克多Agent游戏 - 手牌阶段
主程序入口
"""

from langgraph.graph import StateGraph, END
from game_nodes import (

    GameState, dealer_node, agent_laoying_action, 
    agent_xiaolajiao_action, agent_laoqian_action,
    judge_winner_node, check_hand_end
)
from agent_config import AGENT_PROFILES 
from game_display import print_game_state


def build_game_graph():
    """构建游戏图"""
    graph = StateGraph(GameState)
    
    graph.add_node("dealer", dealer_node)
    graph.add_node("老鹰", agent_laoying_action)
    graph.add_node("小辣椒", agent_xiaolajiao_action)
    graph.add_node("老钱", agent_laoqian_action)
    graph.add_node("judge", judge_winner_node)
    
    graph.set_entry_point("dealer")
    
    graph.add_edge("dealer", "老鹰")
    graph.add_edge("老鹰", "小辣椒")
    graph.add_edge("小辣椒", "老钱")
    
    graph.add_conditional_edges(
        "老钱",
        check_hand_end,
        {
            "judge": "judge",
            "continue": "老鹰",
        },
    )
    
    graph.add_edge("judge", END)
    
    return graph.compile()


def run_game():
    """运行游戏"""
    print("=" * 70)
    print("  德州扑克多Agent游戏 - 手牌阶段")
    print("  基于 LangGraph State 架构")
    print("  模型：通义千问 qwen-plus")
    print("=" * 70)
    print("\n三位玩家：")
    for name, profile in AGENT_PROFILES.items():
        print(f"  - {name} : {profile['描述']}")
    
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
    
    app = build_game_graph()
    
    for hand_num in range(1, 4):
        initial_state["hand_number"] = hand_num
        print_game_state(initial_state)
        
        final_state = app.invoke(initial_state)
        
        initial_state["chips"] = final_state["chips"]
        initial_state["players_info"] = {}
        initial_state["pot"] = 0
        initial_state["messages"] = []
        
        if hand_num < 3:
            user_input = input("\n按 Enter 继续下一手，输入 q 退出：").strip().lower()
            if user_input == "q":
                break
    
    print("\n" + "=" * 70)
    print("  游戏结束！")
    print("  最终筹码：")
    for name in ["老鹰", "小辣椒", "老钱"]:
        print(f"    {name}：{final_state['chips'][name]} 筹码")
    print("=" * 70)


if __name__ == "__main__":
    run_game()
