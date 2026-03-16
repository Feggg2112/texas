"""AI 决策节点：调用阿里千问 LLM，支持多 Agent 并行决策。"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
import asyncio
from typing import Literal
from pydantic import BaseModel, Field, field_validator
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import SystemMessage, HumanMessage
from prompts.poker_prompts import (
    build_decision_prompt,
    PLAYER_SYSTEM_PROMPTS,
    DEFAULT_SYSTEM_PROMPT,
)
from utils.poker_utils import evaluate_hand, hand_strength_preflop


# ── Pydantic 模型：定义 LLM 决策输出的结构 ──────────────────────────────────
# 【教学重点】
# Pydantic 的作用：用「类型声明」替代手写的 JSON 解析 + 校验逻辑
# - BaseModel: 数据类，字段类型即校验规则
# - Field: 给字段加默认值、描述、约束
# - field_validator: 自定义字段校验逻辑
# - model_validate: 从字典/JSON 解析并校验（替代 json.loads + 手动校验）

class DecisionOutput(BaseModel):
    """AI 决策的结构化输出。
    
    Pydantic 会自动：
    1. 校验 action 必须是合法值（否则抛 ValidationError）
    2. 把 amount 强制转为 int（LLM 有时输出字符串 "100"）
    3. 给缺失字段填默认值
    """
    thought: str = Field(default="", description="AI 的思考过程")
    action: Literal["fold", "check", "call", "raise"] = Field(
        default="fold",
        description="行动类型"
    )
    amount: int = Field(default=0, ge=0, description="加注总额，非 raise 时为 0")

    @field_validator("action", mode="before")
    @classmethod
    def normalize_action(cls, v: str) -> str:
        """容错：把 LLM 可能输出的变体（如 'FOLD', 'Check'）统一小写。"""
        if isinstance(v, str):
            v = v.lower().strip()
        # 常见别名映射
        aliases = {"pass": "check", "bet": "raise", "allin": "raise", "all-in": "raise"}
        return aliases.get(v, v)

    @field_validator("amount", mode="before")
    @classmethod
    def coerce_amount(cls, v) -> int:
        """容错：把字符串金额 '100' 转为整数 100。"""
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0


def _parse_llm_response(text: str) -> dict:
    """
    从 LLM 返回文本中提取并验证 JSON 决策块。
    
    【教学重点：为什么用 Pydantic 而不是手写 json.loads？】
    
    手写方式的问题：
        data = json.loads(text)          # 可能 KeyError
        action = data.get("action", "")  # 可能是非法值如 "FOLD"
        amount = data.get("amount", 0)   # 可能是字符串 "100" 而不是整数
        if action not in VALID_ACTIONS:  # 还要手写校验
            action = "fold"
    
    Pydantic 方式：
        DecisionOutput.model_validate(data)  # 一行完成解析+校验+类型转换
        # 自动处理非法值、类型转换、默认值
    
    降级策略（fallback chain）：
    1. 直接解析整个文本
    2. 提取 ```json ... ``` 代码块
    3. 提取第一个 { ... } 对象
    4. 全部失败 → 返回默认弃牌
    """
    import json, re

    def _try_parse(raw: str) -> dict | None:
        try:
            data = json.loads(raw.strip())
            # model_validate 解析并校验，返回 Pydantic 模型实例
            # .model_dump() 转回普通字典
            return DecisionOutput.model_validate(data).model_dump()
        except Exception:
            return None

    # 策略1：直接解析
    result = _try_parse(text)
    if result:
        return result

    # 策略2：提取代码块
    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if match:
        result = _try_parse(match.group(1))
        if result:
            return result

    # 策略3：提取第一个 JSON 对象
    match = re.search(r"\{[\s\S]+\}", text)
    if match:
        result = _try_parse(match.group(0))
        if result:
            return result

    # 策略4：全部失败，返回安全默认值
    return DecisionOutput(thought="解析失败，默认弃牌", action="fold", amount=0).model_dump()


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


# ── LLM 工厂 ──────────────────────────────────────────────────────────────────

def _get_llm(model: str = "qwen-max", temperature: float = 0.4) -> ChatTongyi:
    """返回千问 LLM 实例。API key 从环境变量 DASHSCOPE_API_KEY 读取。"""
    return ChatTongyi(model=model, temperature=temperature)

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
            "action": "fold",
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
