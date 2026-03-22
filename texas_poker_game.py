# -*- coding: utf-8 -*-
"""
德州扑克多Agent游戏
主程序入口
"""

from langgraph.graph import StateGraph, END
from game_nodes import (
    GameState, dealer_node, player_action_node,
    deal_community_node, judge_winner_node, route_player_action
)
from agent_config import AGENT_PROFILES
from game_display import print_game_state


# 玩家顺序（在此添加/删除玩家，其余代码无需改动）
PLAYER_ORDER = ["老鹰", "小辣椒", "老钱", "火线", "铁拳"]


def build_game_graph():
    """构建游戏图"""
    graph = StateGraph(GameState)

    graph.add_node("dealer", dealer_node)
    graph.add_node("player_action", player_action_node)
    graph.add_node("deal_community", deal_community_node)
    graph.add_node("judge", judge_winner_node)

    graph.set_entry_point("dealer")
    graph.add_edge("dealer", "player_action")

    graph.add_conditional_edges(
        "player_action",
        route_player_action,
        {
            "continue": "player_action",
            "deal_community": "deal_community",
            "judge": "judge",
        },
    )

    graph.add_edge("deal_community", "player_action")
    graph.add_edge("judge", END)

    return graph.compile()


def run_game():
    """运行游戏"""
    print("=" * 70)
    print("  德州扑克多Agent游戏")
    print("  基于 LangGraph State 架构")
    print("  模型：通义千问 qwen-plus") 
    print("=" * 70)
    print( str(len(PLAYER_ORDER)) + "位玩家：")
    for name in PLAYER_ORDER:
        profile = AGENT_PROFILES.get(name)
        if profile:
            print("  - " + name + " : " + profile["描述"])
        else:
            print("  - " + name + " : 真人玩家")

    initial_state: GameState = {
        "hand_number": 1,
        "chips": {name: 1000 for name in PLAYER_ORDER},
        "pot": 0,
        "community_cards": [],
        "deck": [],
        "players_info": {name: {"is_human": False} for name in PLAYER_ORDER},
        "messages": [],
        "current_player": PLAYER_ORDER[0],
        "player_order": PLAYER_ORDER,
        "stage": "preflop",
        "hand_ended": False,
        "winner": "",
        "street_finished": False,
    }

    app = build_game_graph()

    for hand_num in range(1, 4):
        initial_state["hand_number"] = hand_num
        print_game_state(initial_state)

        final_state = app.invoke(initial_state)

        initial_state["chips"] = final_state["chips"]
        initial_state["players_info"] = {name: {"is_human": False} for name in PLAYER_ORDER}
        initial_state["pot"] = 0
        initial_state["messages"] = []
        initial_state["hand_ended"] = False
        initial_state["street_finished"] = False
        initial_state["community_cards"] = []
        initial_state["deck"] = []
        initial_state["stage"] = "preflop"

        if hand_num < 3:
            user_input = input("\n按 Enter 继续下一手，输入 q 退出：").strip().lower()
            if user_input == "q":
                break

    print("\n" + "=" * 70)
    print("  游戏结束！")
    print("  最终筹码：")
    for name in PLAYER_ORDER:
        print("    " + name + "：" + str(final_state["chips"][name]) + " 筹码")
    print("=" * 70)

def export_graph_png(output_path: str = "game_graph.png"):
    app = build_game_graph()
    png_bytes = app.get_graph().draw_mermaid_png()
    with open(output_path, "wb") as f:
        f.write(png_bytes)
    print("图已导出：", output_path)


if __name__ == "__main__":
    export_graph_png("game_graph.png")
    run_game()
