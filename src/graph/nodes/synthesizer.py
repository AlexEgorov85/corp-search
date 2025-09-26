# src/graph/nodes/synthesizer.py
from typing import Dict, Any
import logging
from src.agents.registry import AgentRegistry

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


def synthesizer_node(state: Dict[str, Any], agent_registry: AgentRegistry) -> Dict[str, Any]:
    try:
        if state.get("final_answer") is not None:
            return {}

        inst = agent_registry.instantiate_agent("SynthesizerAgent", control=True)
        step_outputs = state.get("step_outputs") or {}
        plan = state.get("plan") or []

        try:
            raw = inst.execute({"plan": plan, "step_outputs": step_outputs, "question": state.get("question")}, context={})
        except TypeError:
            raw = inst.execute({"plan": plan, "step_outputs": step_outputs, "question": state.get("question")})

        if isinstance(raw, dict) and raw.get("status") == "ok":
            final = raw.get("content")
            structured = raw.get("structured") or {}
            LOG.info("synthesizer_node: SynthesizerAgent produced final answer")
            return {"synth_output": structured, "final_answer": final, "finished": True}

        if isinstance(raw, dict):
            final = raw.get("final_answer") or raw.get("content") or str(raw)
            LOG.info("synthesizer_node: SynthesizerAgent returned dict -> using fallback fields")
            return {"synth_output": raw, "final_answer": final, "finished": True}

        # fallback
        final_text = None
        for s in reversed(plan):
            sid = s.get("id")
            out = step_outputs.get(sid)
            if isinstance(out, dict) and out.get("status") == "ok" and out.get("content"):
                final_text = out.get("content")
                break

        if not final_text:
            parts = []
            for sid, out in step_outputs.items():
                if isinstance(out, dict):
                    parts.append(out.get("content") or str(out.get("structured", "")))
                else:
                    parts.append(str(out))
            final_text = "\n".join(p for p in parts if p) or "Нет результатов"

        synth_output = {"final_answer": final_text, "reasoning": "fallback"}
        LOG.info("synthesizer_node: fallback final_answer prepared")
        return {"synth_output": synth_output, "final_answer": final_text, "finished": True}

    except Exception as e:
        LOG.exception("synthesizer_node unexpected error: %s", e)
        return {"synth_output": {"status": "error", "error": str(e)}, "final_answer": "Ошибка синтеза", "finished": True}