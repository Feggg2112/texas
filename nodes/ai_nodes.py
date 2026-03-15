"""AI 决策节点：调用阿里千问 LLM，支持多 Agent 并行决策。"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import re
import asyncio
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import SystemMessage, HumanMessage
from prompts.poker_prompts import (
    build_decision_prompt,
    PLAYER_SYSTEM_PROMPTS,
    DEFAULT_SYSTEM_PROMPT,
)
from utils.poker_utils import evaluate_hand, hand_strength_preflop


# ── LLM 工厂 ──────────────────────────────────────────────────────────────────

def _get_llm(model: str = "qwen-max", temperature: float = 0.4) -> ChatTongyi:
    """
    返回千问 LLM 实例。
    API key 从环境变量 DASHSCOPE_API_KEY 读取。
    """
    return ChatTongyi(
        model=model,
        temperature=temperature,
    )


# ── JSON 解析 ─────────────────────────────────────────────────────────────────

def _parse_llm_response(text: str) -> dict:
    """
    从 LLM 返回文本中提取 JSON 决策块。
    容错：即使 LLM 在 JSON 外输出了额外文字也能解析。
    """
    # 尝试直接解析
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 提取 ```json ... ``` 代码块
    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 提取第一个 { ... } 块
    match = re.search(r"\{[\s\S]+\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # 最终回退：弃牌
    return {"thought": "解析失败，默认弃牌", "action": "fold", "amount": 0}


def _validate_action(decision: dict, state: dict, player_state: dict) -> dict:
    """
    验证并修正 LLM 返回的行动，防止非法操作（如超额加注、无效 check 等）。
    """
    action = decision.get("action", "fold")
    amount = int(decision.get("amount", 0))
    current_bet = state.get("current_bet", 0)
    my_street_bet = player_state.get("current_street_bet", 0)
    my_chips = player_state.get("chips", 0)
    min_raise = state.get("min_raise", state.get("big_blind", 20))
    call_amount = max(0, current_bet - my_street_bet)

    if action == "check" and call_amount > 0:
        # 有人下注时不能过牌，改为跟注
        action = "call"
        amount = 0
    elif action == "raise":
        # 保证加注额合法
        min_total = current_bet + min_raise
        max_total = my_street_bet + my_chips  # all-in 上限
        amount = max(min_total, amount)
        amount = min(amount, max_total)
    else:
        amount = 0

    decision["action"] = action
    decision["amount"] = amount
    return decision


# ── 单个 AI 玩家决策（同步） ──────────────────────────────────────────────────

def _single_ai_decide(player_state: dict, game_state: dict, llm: ChatTongyi) -> dict:
    """
    为单个 AI 玩家调用 LLM 做出决策。
    返回 {'thought': str, 'action': str, 'amount': int}
    """
    community = game_state.get("community_cards", [])
    hole_cards = player_state.get("hole_cards", [])

    # 手牌强度评估
    if len(community) >= 3:
        hand_eval = evaluate_hand(hole_cards, community)
    else:
        strength = hand_strength_preflop(hole_cards)
        hand_eval = {
            "strength": strength,
            "rank_string": "翻前评估",
            "score": int((1 - strength) * 7461) + 1,
            "rank_class": 9,
        }

    # 构建提示词
    prompt = build_decision_prompt(player_state, game_state, hand_eval)

    # 选择对应人设的 system prompt
    player_type = player_state.get("player_type", "balanced")
    system_prompt = PLAYER_SYSTEM_PROMPTS.get(player_type, DEFAULT_SYSTEM_PROMPT)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt),
    ]

    try:
        response = llm.invoke(messages)
        raw_text = response.content
        decision = _parse_llm_response(raw_text)
        decision = _validate_action(decision, game_state, player_state)
    except Exception as e:
        decision = {
            "thought": f"LLM 调用失败: {e}，默认跟注",
            "action": "call",
            "amount": 0,
        }

    return decision


# ── 单玩家决策节点（供 LangGraph 使用） ──────────────────────────────────────

def ai_decision_node(state: dict) -> dict:
    """
    LangGraph 节点：为当前 AI 玩家做出决策。
    写入 state['agent_decision'] 和 state['agent_thoughts']。
    """
    players = state["players"]
    idx = state["current_player_index"]
    player_state = players[idx]

    llm = _get_llm()
    decision = _single_ai_decide(player_state, state, llm)

    # 更新上帝视角思考记录
    thoughts = dict(state.get("agent_thoughts") or {})
    thoughts[player_state["name"]] = decision.get("thought", "")

    return {
        "agent_decision": decision,
        "agent_thoughts": thoughts,
    }


# ── 批量并行决策（可选：一次性让所有 AI 都想好，用于观战模式） ────────────────

async def _async_decide(player_state: dict, game_state: dict, llm: ChatTongyi) -> tuple[str, dict]:
    """异步包装单个决策，返回 (player_name, decision)。"""
    loop = asyncio.get_event_loop()
    decision = await loop.run_in_executor(None, _single_ai_decide, player_state, game_state, llm)
    return player_state["name"], decision


def batch_ai_decisions(state: dict) -> dict:
    """
    并行调用所有 AI 玩家的 LLM 决策（适用于需要预先收集所有思路的场景）。
    返回更新后的 agent_thoughts 字典。
    """
    players = state["players"]
    ai_players = [
        p for p in players
        if p["is_active"] and p["is_ai"] and not p["is_all_in"]
    ]

    if not ai_players:
        return {"agent_thoughts": state.get("agent_thoughts", {})}

    llm = _get_llm()

    async def _run_all():
        tasks = [_async_decide(p, state, llm) for p in ai_players]
        return await asyncio.gather(*tasks)

    results = asyncio.run(_run_all())
    thoughts = dict(state.get("agent_thoughts") or {})
    for name, decision in results:
        thoughts[name] = decision.get("thought", "")

    return {"agent_thoughts": thoughts}
