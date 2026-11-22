"""Public literature-review tasks; this module cannot read reference answers."""
import json
import re
from .records import fingerprint

SYSTEM = "Use the supplied research material. Follow the current task. Do not invent findings or citations."


def document(paper):
    return paper["title"] + "\n\n" + "\n\n".join(
        f"[{p['id']}] {p['section']}\n{p['text']}" for p in paper["paragraphs"])


def reference_address_audit(paper, background_text, *, origin):
    """Observe addresses in the actual background, never infer or repair evidence.

    A valid ID mention is not a verified quotation or a useful evidence mapping.
    No mentions can also mean legitimate abstention: this is not a delivery gate.
    For C2C, text describes the source document, not decoded receiver knowledge.
    """
    known = {p["id"] for p in paper["paragraphs"]}
    mentions = list(dict.fromkeys(re.findall(r"(?<!\w)p[0-9]+(?!\w)", background_text)))
    unknown = [p for p in mentions if p not in known]
    warnings = []
    if not mentions:
        warnings.append("no_paragraph_addresses_in_background; inspect handoff or explicit abstention")
    if unknown:
        warnings.append("unknown_paragraph_addresses; no host repair")
    return {"origin": origin, "source_paper_fingerprint": fingerprint(paper),
            "background_text_fingerprint": fingerprint(background_text),
            "known_id_mentions": [p for p in mentions if p in known],
            "unknown_id_mentions": unknown, "warnings": warnings,
            "semantic_sufficiency_verified": False, "dispatch_gate": False,
            "catalog_in_command_is_not_evidence_mapping": True}


def plan_request(paper, order="paper_first"):
    if order not in {"paper_first", "goal_first"}:
        raise ValueError("Unknown public block order")
    introduction = "Prepare an evidence-backed review of this paper for the questions below."
    body = "\n\nPAPER\n" + document(paper)
    goal = ("\n\nQUESTIONS\n" + json.dumps(paper["questions"], ensure_ascii=False)
        + "\nDispatch one evidence search per question, in a useful order. For each, give its question_id "
          "and a short search focus (section, concept, or relation to inspect). The runtime will retrieve "
          "the selected original paragraphs, then assign an answer writer with access only to that evidence. "
          "Return {\"tasks\":[{\"question_id\":\"q1\",\"focus\":\"...\"}, ...]}. "
          "Cover each supplied question exactly once. Do not answer the questions or repeat the paper.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content":
        introduction + (body + goal if order == "paper_first" else goal + body)}]


def plan_schema(paper):
    return {"type": "object", "properties": {"tasks": {"type": "array", "minItems": len(paper["questions"]),
        "maxItems": len(paper["questions"]), "items": {"type": "object", "properties": {
            "question_id": {"type": "string", "enum": [q["id"] for q in paper["questions"]]},
            "focus": {"type": "string", "minLength": 1, "maxLength": 240}},
            "required": ["question_id", "focus"], "additionalProperties": False}}},
        "required": ["tasks"], "additionalProperties": False}


def parse_plan(output, paper):
    value = json.loads(output)
    expected = {q["id"] for q in paper["questions"]}
    if set(value) != {"tasks"} or len(value["tasks"]) != len(expected):
        raise ValueError("Plan must cover the actual question set")
    seen = set()
    for task in value["tasks"]:
        if set(task) != {"question_id", "focus"} or task["question_id"] not in expected - seen:
            raise ValueError("Unknown or duplicated planned question")
        if not isinstance(task["focus"], str) or not 1 <= len(task["focus"]) <= 240:
            raise ValueError("Invalid search focus")
        seen.add(task["question_id"])
    return value["tasks"]


def evidence_schema(paper):
    return {"type": "object", "properties": {"paragraphs": {"type": "array", "minItems": 0,
        "maxItems": 6, "items": {"type": "string", "enum": [p["id"] for p in paper["paragraphs"]]}}},
        "required": ["paragraphs"], "additionalProperties": False}


def search_command(question, focus):
    return ("Locate the original paper paragraphs needed to answer this question: " + question
        + "\nSearch focus: " + focus + "\nReturn only {\"paragraphs\":[\"p0\", ...]} with up to six "
          "original paragraph IDs, without duplicates. Choose evidence, not merely a repeated question. "
          "An empty list is permitted when you find no supporting evidence.")


def answer_command(question, evidence):
    return ("Answer the research question briefly using only the retrieved passages below. "
        "If these passages do not support an answer, return unanswerable. Return the answer only.\n\n"
        "Question: " + question + "\n\nRetrieved passages:\n" + "\n\n".join(
            f"[{p['id']}] {p['text']}" for p in evidence))


def brief_command(paper):
    return ("Prepare a practical handoff for evidence-search workers answering these questions:\n"
        + json.dumps(paper["questions"], ensure_ascii=False)
        + "\nYou may identify candidate evidence, quote passages, explain relations, or give useful answers. "
          "Keep the original paragraph IDs so workers can request the original passages. "
          "They will receive their own question and the full list of valid IDs, but cannot read the full paper "
          "except through this handoff. Preserve what they need. Do not invent IDs. Be concise and complete.")
