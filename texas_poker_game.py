# -*- coding: utf-8 -*-
"""
德州扑克多Agent游戏
主程序入口
"""

from langgraph.graph import StateGraph, END
from game_nodes import (
    GameState,
    dealer_node,
    player_action_node,
    human_action_node,
    deal_community_node,
    judge_winner_node,
    route_player_action,
    route_entry_action_node,
)
from agent_config import AGENT_PROFILES
from game_display import print_game_state


AI_ONLY_ORDER = ["老鹰", "小辣椒", "老钱", "火线", "铁拳"]
BATTLE_ORDER = ["你", "老鹰", "小辣椒", "老钱", "火线", "铁拳"]
HUMAN_PLAYER_NAME = "你"


def _base_players_info(player_order, human_enabled: bool):
    return {
        name: {"is_human": human_enabled and name == HUMAN_PLAYER_NAME}
        for name in player_order
    }


def choose_mode() -> str:
    print("\n请选择模式：")
    print("  1) 上帝视角模式（纯AI对战，显示所有手牌）")
    print("  2) 对战模式（你参与，隐藏AI手牌，摊牌时公开）")
    while True:
        choice = input("请输入 1 或 2：").strip()
        if choice == "1":
            return "god"
        if choice == "2":
            return "battle"
        print("输入无效，请输入 1 或 2。")


def action_router_node(_state: GameState) -> dict:
    """仅用于分流到人类或AI行动节点"""
    return {}


def build_game_graph():
    """构建游戏图"""
    graph = StateGraph(GameState)

    graph.add_node("dealer", dealer_node)
    graph.add_node("action_router", action_router_node)
    graph.add_node("player_action", player_action_node)
    graph.add_node("human_action", human_action_node)
    graph.add_node("deal_community", deal_community_node)
    graph.add_node("judge", judge_winner_node)

    graph.set_entry_point("dealer")
    graph.add_edge("dealer", "action_router")

    graph.add_conditional_edges(
        "action_router",
        route_entry_action_node,
        {
            "player_action": "player_action",
            "human_action": "human_action",
        },
    )

    graph.add_conditional_edges(
        "player_action",
        route_player_action,
        {
            "continue": "action_router",
            "deal_community": "deal_community",
            "judge": "judge",
        },
    )

    graph.add_conditional_edges(
        "human_action",
        route_player_action,
        {
            "continue": "action_router",
            "deal_community": "deal_community",
            "judge": "judge",
        },
    )

    graph.add_edge("deal_community", "action_router")
    graph.add_edge("judge", END)

    return graph.compile()


def run_game():
    """运行游戏（锦标赛模式）"""
    print("=" * 70)
    print("  德州扑克多Agent游戏")
    print("  基于 LangGraph State 架构")
    print("  模型：通义千问 qwen-plus")
    print("=" * 70)

    mode = choose_mode()
    is_battle = mode == "battle"
    player_order = BATTLE_ORDER if is_battle else AI_ONLY_ORDER

    print("\n当前模式：" + ("对战模式" if is_battle else "上帝视角模式"))
    print(str(len(player_order)) + "位玩家：")
    for name in player_order:
        profile = AGENT_PROFILES.get(name)
        if profile:
            print("  - " + name + " : " + profile["描述"])
        else:
            print("  - " + name + " : 真人玩家")

    initial_state: GameState = {
        "hand_number": 1,
        "chips": {name: 2000 for name in player_order},
        "pot": 0,
        "community_cards": [],
        "deck": [],
        "players_info": _base_players_info(player_order, is_battle),
        "messages": [],
        "current_player": player_order[0],
        "player_order": player_order,
        "stage": "preflop",
        "hand_ended": False,
        "winner": "",
        "street_finished": False,
        "ante": 80,
        "min_open_raise": 100,
        "max_raises_per_street": 3,
        "raises_in_street": 0,
        "last_raise_size": 100,
        "mode": mode,
        "human_player": HUMAN_PLAYER_NAME,
        "show_hole_cards": (mode == "god"),
    }

    app = build_game_graph()

    hand_num = 1
    final_state = initial_state
    while True:
        alive_players = [n for n in player_order if initial_state["chips"].get(n, 0) > 0]
        if len(alive_players) <= 1:
            break

        initial_state["hand_number"] = hand_num
        print_game_state(initial_state)

        final_state = app.invoke(initial_state)

        initial_state["chips"] = final_state["chips"]
        initial_state["players_info"] = _base_players_info(player_order, is_battle)
        initial_state["pot"] = 0
        initial_state["messages"] = []
        initial_state["hand_ended"] = False
        initial_state["street_finished"] = False
        initial_state["community_cards"] = []
        initial_state["deck"] = []
        initial_state["stage"] = "preflop"
        initial_state["raises_in_street"] = 0
        initial_state["last_raise_size"] = initial_state.get("min_open_raise", 100)

        hand_num += 1

        alive_after_hand = [n for n in player_order if initial_state["chips"].get(n, 0) > 0]
        if len(alive_after_hand) <= 1:
            break

        user_input = input("\n按 Enter 继续下一手，输入 q 退出：").strip().lower()
        if user_input == "q":
            break

    print("\n" + "=" * 70)
    print("  游戏结束！")
    winner_names = [n for n in player_order if final_state["chips"].get(n, 0) > 0]
    if len(winner_names) == 1:
        print("  冠军：" + winner_names[0])
    print("  最终筹码：")
    for name in player_order:
        print("    " + name + "：" + str(final_state["chips"][name]) + " 筹码")
    print("=" * 70)


def export_graph_png(output_path: str = "game_graph.png"):
    app = build_game_graph()
    png_bytes = app.get_graph().draw_mermaid_png()
    with open(output_path, "wb") as f:
        f.write(png_bytes)
    print("图已导出：", output_path)


if __name__ == "__main__":
    # export_graph_png("game_graph.png")
    run_game()
