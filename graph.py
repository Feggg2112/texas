# -*- coding: utf-8 -*-
"""LangGraph 图定义：德州扑克完整状态机。"""
from __future__ import annotations
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from state import GameState
from nodes.game_nodes import (
    initialize_game,
    deal_community_cards,
    execute_action,
    advance_street,
    showdown,
    check_game_over,
    route_after_action,
    route_after_street,
    route_game_over,
    route_current_player,
)
from nodes.ai_nodes import ai_decision_node
from nodes.chat_nodes import (
    chat_start,
    ai_chat_node,
    human_chat_node,
    route_chat,
)



def human_input_node(state: dict) -> dict:
    """
    LangGraph interrupt 节点：暂停图执行，等待人类玩家输入。
    外部通过 graph.update_state(config, {'human_action': {...}}) 写入行动后继续。
    """
    players = state["players"]
    idx = state["current_player_index"]
    player = players[idx]

    action = interrupt({
        "waiting_for": player["name"],
        "player_id": player["id"],
        "hole_cards": player["hole_cards"],
        "community_cards": state["community_cards"],
        "pot": state["pot"],
        "current_bet": state["current_bet"],
        "my_chips": player["chips"],
        "call_amount": max(0, state["current_bet"] - player["current_street_bet"]),
        "min_raise": state["min_raise"],
    })

    return {"human_action": action}



def build_graph(checkpointer=None):
    """
    构建并编译完整的德州扑克 LangGraph。

    完整流程：
      initialize_game
          |1
      deal_cards
          |1
      chat_start          <- 每个街道开始前重置对话轮次
          |
      route_chat          <- 循环：还有玩家未发言？
       /       \
    ai_chat  human_chat   <- 每人依次发言（或沉默）
       \       /
      route_chat (循环)
          | chat_done
      route_player        <- 判断当前玩家是 AI 还是人类
       /         \
    ai_decision  human_input
       \ 1        /1
      execute_action
    """
    builder = StateGraph(GameState)

    # ── 注册节点
    builder.add_node("initialize_game", initialize_game)
    builder.add_node("deal_cards", deal_community_cards)
    # 对话阶段节点
    builder.add_node("chat_start", chat_start)
    builder.add_node("ai_chat", ai_chat_node)
    builder.add_node("human_chat", human_chat_node)
    # 下注阶段节点
    builder.add_node("route_player", lambda s: s)  # 纯路由节点，不修改状态
    builder.add_node("ai_decision", ai_decision_node)
    builder.add_node("human_input", human_input_node)
    builder.add_node("execute_action", execute_action)
    builder.add_node("advance_street", advance_street)
    builder.add_node("showdown", showdown)
    builder.add_node("check_game_over", check_game_over)

    # ── 入口
    builder.set_entry_point("initialize_game")

    # ── 固定边
    builder.add_edge("initialize_game", "deal_cards")
    builder.add_edge("deal_cards", "chat_start")
    builder.add_edge("ai_decision", "execute_action")
    builder.add_edge("human_input", "execute_action")
    builder.add_edge("showdown", "check_game_over")

    # ── 条件边0：chat_start 后进入对话循环
    builder.add_conditional_edges(
        "chat_start",
        route_chat,
        {
            "ai_chat": "ai_chat",
            "human_chat": "human_chat",
            "chat_done": "route_player",
        },
    )

    # ── 条件边1：每次发言后再判断下一个发言者
    builder.add_conditional_edges(
        "ai_chat",
        route_chat,
        {
            "ai_chat": "ai_chat",
            "human_chat": "human_chat",
            "chat_done": "route_player",
        },
    )
    builder.add_conditional_edges(
        "human_chat",
        route_chat,
        {
            "ai_chat": "ai_chat",
            "human_chat": "human_chat",
            "chat_done": "route_player",
        },
    )

    # ── 条件边2：当前玩家是 AI 还是人类
    builder.add_conditional_edges(
        "route_player",
        route_current_player,
        {
            "ai_decision": "ai_decision",
            "human_input": "human_input",
        },
    )

    # ── 条件边3：execute_action 后分支
    builder.add_conditional_edges(
        "execute_action",
        route_after_action,
        {
            "continue_betting": "route_player",
            "street_complete": "advance_street",
            "only_one_active": "showdown",
        },
    )

    # ── 条件边4：advance_street 后分支
    builder.add_conditional_edges(
        "advance_street",
        route_after_street,
        {
            "deal_cards": "deal_cards",
            "showdown": "showdown",
        },
    )

    # ── 条件边5：游戏结束检查
    builder.add_conditional_edges(
        "check_game_over",
        route_game_over,
        {
            "new_round": "initialize_game",
            "end": END,
        },
    )

    checkpointer = checkpointer or MemorySaver()
    return builder.compile(checkpointer=checkpointer)



def make_initial_state(
    player_configs: list[dict],
    small_blind: int = 10,
    big_blind: int = 20,
) -> dict:
    """
    构造传入图的初始 state。

    player_configs 示例:
    [
        {'name': 'Alice', 'chips': 1000, 'player_type': 'aggressive', 'is_ai': True},
        {'name': '你',    'chips': 1000, 'player_type': 'balanced',   'is_human': True},
    ]
    """
    players = []
    for i, cfg in enumerate(player_configs):
        players.append({
            "id": i,
            "name": cfg["name"],
            "chips": cfg.get("chips", 1000),
            "hole_cards": [],
            "is_active": True,
            "is_all_in": False,
            "is_human": cfg.get("is_human", False),
            "is_ai": cfg.get("is_ai", True),
            "current_street_bet": 0,
            "total_bet": 0,
            "position": i,
            "player_type": cfg.get("player_type", "balanced"),
            "vpip": 0.0,
            "pfr": 0.0,
            "thought": "",
        })

    return {
        "players": players,
        "deck": [],
        "community_cards": [],
        "pot": 0,
        "side_pots": [],
        "current_bet": 0,
        "min_raise": big_blind,
        "dealer_position": -1,
        "small_blind": small_blind,
        "big_blind": big_blind,
        "current_player_index": 0,
        "action_order": [],
        "street": "preflop",
        "street_action_count": 0,
        "action_history": [],
        "chat_history": [],
        "pending_chat": [],
        "chat_round_index": 0,
        "agent_thoughts": {},
        "agent_decision": None,
        "human_action": None,
        "human_chat": None,
        "round_number": 0,
        "game_over": False,
        "winners": [],
        "winner_message": "",
        "error_message": "",
    }
