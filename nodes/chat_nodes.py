"""对话节点：在每个街道行动前，让玩家发言（可欺骗、可沉默）。

【对话流程设计】

每条街道（preflop/flop/turn/river）进入下注阶段前，先走一轮对话阶段：

  deal_cards
      ↓
  chat_start         ← 重置对话轮次计数
      ↓
  route_chat         ← 判断还有没有玩家需要发言
      ↓              ↓
  ai_chat      human_chat (interrupt)
      ↓              ↓
  chat_next          ← 推进到下一个发言玩家
      ↓
  route_player       ← 对话结束，进入下注阶段

【欺骗机制】
- AI 发言由 LLM 根据人设生成，is_bluff 字段记录是否为欺骗性发言
- 发言内容对所有玩家可见，is_bluff 和 inner_reason 只在上帝视角显示
- 决策 prompt 里会注入本街道的对话历史，让 LLM "听进去" 并受影响
"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import re
from langgraph.types import interrupt
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import SystemMessage, HumanMessage
from prompts.poker_prompts import (
    build_chat_prompt,
    PLAYER_SYSTEM_PROMPTS,
    DEFAULT_SYSTEM_PROMPT,
    CHAT_SYSTEM_PROMPT_ADDON,
)
from utils.poker_utils import evaluate_hand, hand_strength_preflop


def _get_llm() -> ChatTongyi:
    return ChatTongyi(model="qwen-max", temperature=0.7)  # 对话用更高温度，更有创意


def _parse_chat_response(text: str) -> dict:
    """从 LLM 返回中提取对话 JSON，容错处理。"""
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{[\s\S]+\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    # 回退：沉默
    return {"message": "", "is_bluff": False, "inner_reason": "解析失败，默认沉默"}


def _get_hand_eval(player_state: dict, game_state: dict) -> dict:
    """获取当前手牌强度评估，复用 ai_nodes 的逻辑。"""
    community = game_state.get("community_cards", [])
    hole_cards = player_state.get("hole_cards", [])
    if len(community) >= 3:
        return evaluate_hand(hole_cards, community)
    strength = hand_strength_preflop(hole_cards)
    return {
        "strength": strength,
        "rank_string": "翻前评估",
        "score": int((1 - strength) * 7461) + 1,
        "rank_class": 9,
    }


# ── 节点1：chat_start ─────────────────────────────────────────────────────────

def chat_start(state: dict) -> dict:
    """
    对话阶段开始：重置轮次计数，清空本轮待收集缓冲。
    每条街道发完牌后、进入下注循环前调用一次。
    """
    return {
        "chat_round_index": 0,
        "pending_chat": [],
    }


# ── 节点2：ai_chat ────────────────────────────────────────────────────────────

def ai_chat_node(state: dict) -> dict:
    """
    AI 玩家发言节点。
    根据人设和当前局面，LLM 生成一句话（可以是欺骗、施压、示弱或沉默）。
    发言写入 chat_history（全局追加）和 pending_chat（本轮缓冲）。
    """
    idx = state["chat_round_index"]
    # 找到本轮发言的玩家（跳过已弃牌的）
    active_players = [p for p in state["players"] if p["is_active"]]
    if idx >= len(active_players):
        return {}  # 防御性处理

    player_state = active_players[idx]
    game_state = state
    hand_eval = _get_hand_eval(player_state, game_state)

    # 构建对话提示词
    prompt = build_chat_prompt(player_state, game_state, hand_eval)
    player_type = player_state.get("player_type", "balanced")
    base_system = PLAYER_SYSTEM_PROMPTS.get(player_type, DEFAULT_SYSTEM_PROMPT)
    # 在原有人设 system prompt 后追加「桌面心理战」补充说明
    system_prompt = base_system + CHAT_SYSTEM_PROMPT_ADDON

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt),
    ]

    try:
        llm = _get_llm()
        response = llm.invoke(messages)
        result = _parse_chat_response(response.content)
    except Exception as e:
        result = {"message": "", "is_bluff": False, "inner_reason": f"LLM 调用失败: {e}"}

    message = result.get("message", "").strip()
    is_bluff = bool(result.get("is_bluff", False))
    inner_reason = result.get("inner_reason", "")

    # 构造消息记录
    chat_entry = {
        "player_name": player_state["name"],
        "player_id": player_state["id"],
        "message": message,
        "is_bluff": is_bluff,          # 上帝视角可见
        "inner_reason": inner_reason,   # 上帝视角可见
        "street": state.get("street", ""),
        "is_silence": message == "",
    }

    pending = list(state.get("pending_chat") or [])
    pending.append(chat_entry)

    return {
        "chat_history": [chat_entry],  # operator.add 追加到全局历史
        "pending_chat": pending,
        "chat_round_index": idx + 1,
    }


# ── 节点3：human_chat ─────────────────────────────────────────────────────────

def human_chat_node(state: dict) -> dict:
    """
    人类玩家发言节点（interrupt 暂停，等待外部输入）。
    外部通过 graph.update_state(config, {'human_chat': '...'}) 写入发言后继续。
    人类可以输入任何文字，或直接回车表示沉默。
    """
    idx = state["chat_round_index"]
    active_players = [p for p in state["players"] if p["is_active"]]
    if idx >= len(active_players):
        return {}

    player_state = active_players[idx]
    street = state.get("street", "")
    recent_chat = [
        m for m in state.get("chat_history", []) if m.get("street") == street
    ]

    raw_input = interrupt({
        "waiting_for_chat": player_state["name"],
        "player_id": player_state["id"],
        "hole_cards": player_state["hole_cards"],
        "community_cards": state["community_cards"],
        "pot": state["pot"],
        "recent_chat": recent_chat,
        "prompt": "输入你想说的话（直接回车=沉默）:",
    })

    message = (raw_input or "").strip()
    chat_entry = {
        "player_name": player_state["name"],
        "player_id": player_state["id"],
        "message": message,
        "is_bluff": False,
        "inner_reason": "(人类玩家)",
        "street": street,
        "is_silence": message == "",
    }

    pending = list(state.get("pending_chat") or [])
    pending.append(chat_entry)

    return {
        "chat_history": [chat_entry],
        "pending_chat": pending,
        "chat_round_index": idx + 1,
        "human_chat": None,
    }


# ── 路由函数 ──────────────────────────────────────────────────────────────────

def route_chat(state: dict) -> str:
    """
    判断对话轮次是否结束。
    活跃玩家都发言（或沉默）过一次后，进入下注阶段。
    否则判断下一个发言者是 AI 还是人类。
    """
    active_players = [p for p in state["players"] if p["is_active"]]
    idx = state.get("chat_round_index", 0)

    if idx >= len(active_players):
        return "chat_done"  # 所有人都说完了，进入下注

    next_player = active_players[idx]
    if next_player["is_human"]:
        return "human_chat"
    return "ai_chat"
 