from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from memory_baseline.core.env import load_project_env
from memory_baseline.core.io import read_predictions, write_predictions
from memory_baseline.core.llm_cache import cached_response, write_cached_response
from memory_baseline.core.token_accounting import add_judge_time, add_judge_tokens, add_query_time, add_query_tokens, new_token_summary
from memory_baseline.core.utils import ensure_dir, estimate_messages_tokens, estimate_tokens, read_jsonl, write_json, write_jsonl
from memory_baseline.data.longmemeval import is_abstention_sample, load_question_ids, load_samples
from memory_baseline.evaluation.error_analysis import write_error_analysis
from memory_baseline.generation.answerer import build_answer_messages as build_baseline_answer_messages
from memory_baseline.generation.judge import get_longmemeval_prompt
from memory_baseline.retrieval.formatter import format_evidence_for_answerer


ANSWER_SYSTEM_PROMPT = """You answer LongMemEval memory questions using only recalled memory evidence.
The user's question date is: {question_date}.
Interpret today, yesterday, last week, current, now, and other relative dates against the question date.
Think through the evidence internally, then output only the final answer.
If the evidence is insufficient, answer exactly: The information is not available in the recalled memory."""


ANSWER_INSTRUCTIONS = {
    "multi-session": "Combine all relevant sessions before answering. For counts, lists, sums, or comparisons, deduplicate repeated mentions and answer from the complete set.",
    "temporal-reasoning": "Order relevant memories by timestamp before answering. For elapsed-time questions, compute against the question date.",
    "knowledge-update": "If memories conflict because information was updated, use the latest relevant update unless the question asks about an earlier state.",
    "single-session-preference": "Use the recalled personal preference or constraint directly; do not give generic advice when a specific preference is available.",
}


MEM0_LME_ANSWER_PROMPT = """You are a personal assistant with access to memories from past conversations with a user. Answer the question using information from the recalled memories below.

IMPORTANT: Today's date is {question_date}. Compute relative time expressions against this date.
IMPORTANT: If the memories contain the numbers, dates, prices, ages, or facts needed to answer, do the computation. Do not abstain when the raw data exists, even if it is scattered across different conversations.
IMPORTANT: Pay attention to the exact entity, role, title, variant, and context in the question. If memories only mention a different entity or context, say the information is not available.
IMPORTANT: Use only the recalled memories. Do not invent numbers, prices, dates, addresses, names, or events.

Before answering, reason step-by-step inside <mem_thinking> tags:
- List every relevant memory, including memories that appear late in the evidence.
- For counting, list, sum, or comparison questions: enumerate every candidate item with date/session context, deduplicate repeated mentions, then compute the final answer.
- For cross-topic computation: identify each needed fact independently and where it appears, then compute.
- For temporal questions: identify the relevant event dates and compute intervals from the question date or from the event reference point asked by the question.
- For knowledge updates or conflicting values: use the latest relevant update unless the question asks about an earlier state.
- For preference or recommendation questions: list the user's relevant preferences, constraints, owned tools/resources, and anti-preferences before selecting the answer.
- Before abstaining, check whether any memory contains a relevant direct or indirect fact. Abstain only when the requested information is genuinely missing or the question asks about a mismatched entity/context.

Rules:
1. Always try to answer if the topic appears in memory, even indirectly.
2. Most recent wins for current status, counts, preferences, or updated facts. Historical event dates should use the memory closest to the event when applicable.
3. For time-bounded questions, compute the date window first, then scan all memories for events in range.
4. For comparison or ordering questions, both compared items must be supported by memory.
5. Actions and completed events beat plans. If a plan has a specific date and no later contradiction, treat it as likely completed on that date.
6. User-stated facts are stronger evidence for personal questions than assistant advice or generic information.
7. Keep the final answer concise and specific.

{evidence_text}

Question: {question}

Work through the instructions in <mem_thinking> tags, then give the final answer after "ANSWER:"."""


GLOBAL_REASONING_ANSWER_PROMPT = """You answer LongMemEval memory questions using only recalled memory evidence.

The user's question date is {question_date}. Interpret relative time expressions against this date.
Use only the recalled memories and the current question. Do not invent unsupported names, numbers, dates, prices, preferences, or events.

Before answering, reason step-by-step inside <mem_thinking> tags:
- Identify what kind of answer the question asks for: direct fact, list, count, comparison, temporal calculation, update/current state, or personalized recommendation.
- List every recalled memory that is relevant to that answer. Do not stop after the first match.
- If the question asks for a personalized recommendation or preference-based response, extract the user's relevant preferences, anti-preferences, constraints, owned tools/resources, and prior interests from the memories. Then tailor the answer to those details and avoid generic suggestions when specific preferences exist.
- If the question asks for a count, list, sum, or comparison, enumerate each candidate item with date/session context, deduplicate repeated mentions, then compute the answer.
- If the question asks about time, identify the relevant dates and compute against the question date or the reference event requested by the question.
- If memories conflict, use the latest relevant update for current-state questions.
- Abstain only when the requested information is genuinely missing or the evidence is about a different entity/context.

Rules:
1. Answer directly and specifically.
2. For preference questions, the final answer may be a useful short recommendation, not just a fact span.
3. For fact questions, keep the final answer concise.
4. If evidence is insufficient, say: The information is not available in the recalled memory.

{evidence_text}

Question: {question}

Work through the instructions in <mem_thinking> tags, then give the final answer after "ANSWER:"."""


TRUEMEMORY_LME_ANSWER_PROMPT = """You are a memory assistant answering questions about past conversations.
You have been given retrieved conversation excerpts as context.

INSTRUCTIONS:
1. Read ALL context carefully; the answer may be spread across multiple excerpts.
2. Look for specific names, dates, numbers, and details.
3. Pay attention to who said what. Speaker attribution matters.
4. For time questions, look for date mentions and temporal references.
5. If a preference, plan, status, or situation changed over time, give the latest supported information unless the question asks for an earlier state.
6. If multiple pieces of evidence exist, synthesize them.
7. For count, list, sum, or comparison questions, consider every relevant excerpt before computing the answer.
8. Give a concise, specific answer.
9. If the context genuinely does not contain the answer, say: The information is not available in the recalled memory.

Current date: {question_date}

Context:
{evidence_text}

Question: {question}

Think step by step, then give the final answer after "ANSWER:"."""


STRUCTURED_REASONING_ANSWER_PROMPT = """You answer LongMemEval memory questions using only recalled memory evidence.

The user's question date is {question_date}. Interpret relative time expressions against this date.
Use only the recalled memories and the current question. Do not invent unsupported names, numbers, dates, prices, preferences, or events.

First build a compact scratchpad in this exact structure:

<analysis_table>
For every recalled memory that may affect the answer, write one row:
- source_time:
- source_session:
- evidence:
- extracted_fact:
- candidate_value:
- include_or_exclude:
- reason:
</analysis_table>

<task_logic>
If the question asks for a count, list, sum, total, amount, or comparison:
1. List all included candidates.
2. Merge duplicate mentions of the same item/event/value.
3. Exclude assistant-only examples, generic advice, hypothetical options, and plans unless the user states completion or a dated plan has no contradiction.
4. Compute the final count, list, sum, or comparison from the included candidates only.

If the question asks about time, order, elapsed duration, latest/current status, or a relative date:
1. Resolve the relevant date window against the question date.
2. List dated events or values.
3. Sort them by date.
4. Use the latest relevant user-stated value for current status unless the question asks for a historical state.

If the question asks for a preference or recommendation:
1. Extract the user's preferences, anti-preferences, constraints, owned tools/resources, and prior interests.
2. Tailor the final answer to those constraints even if the destination or surface topic is new.
</task_logic>

<final_check>
- Every final answer item must appear in included candidates.
- Do not count duplicate mentions twice.
- Do not count generic assistant advice as a user fact.
- If the requested information is genuinely missing or about a mismatched entity/context, answer exactly: The information is not available in the recalled memory.
</final_check>

{evidence_text}

Question: {question}

Work through the structure, then give the final answer after "ANSWER:"."""


MS_EVIDENCE_TABLE_ANSWER_PROMPT = """You answer LongMemEval multi-session memory questions using only recalled memory evidence.

The user's question date is {question_date}. Interpret relative time expressions against this date.
Each evidence row is a recalled memory turn. Do not use outside knowledge.

Before answering, internally build an operand ledger:
- identify every row that may contain a candidate item, event, amount, duration, person, place, or status relevant to the question;
- include only user-specific facts unless the question asks about assistant information;
- exclude generic assistant advice, examples, hypothetical options, and repeated mentions of the same real-world item/event;
- for counts/lists, deduplicate by the real-world item or event;
- for sums/totals, include each distinct supported amount once;
- use stated approximate values such as about, around, nearly, or over when the question asks for a practical total or difference;
- for current-state questions, prefer the latest relevant user-stated status.

Do not abstain when the rows contain direct or approximate numbers, dates, prices, durations, or named events needed to answer.
If the rows genuinely do not contain enough evidence, answer exactly: The information is not available in the recalled memory.
Return only the final answer after "ANSWER:".

{evidence_text}

Question: {question}

ANSWER:"""


GPT41_LEDGER_ANSWER_PROMPT = """You answer LongMemEval memory questions using only recalled memory evidence.

Question date: {question_date}
Use the question date for relative words in the question. Use each memory row's own date for relative words inside that memory row.
Use only the recalled memories. Do not use outside knowledge.

Read all recalled memory rows before deciding. The answer may be in a later row or may require combining rows from different sessions.

Build a brief internal ledger:
1. Identify the answer slot asked by the question, for example who, what, where, when, how many, how much, which item, order, latest/current state, or personalized recommendation.
2. Scan every memory row and extract only facts that could fill that slot.
3. For counts, lists, totals, comparisons, or order questions, include every distinct supported item/event/value, remove duplicates, and then compute.
4. For latest/current/update questions, compare dated evidence and use the latest relevant value unless the question asks for an earlier state.
5. For preference questions, extract the user's platform, genre, style, constraints, anti-preferences, owned resources, and prior successful choices. Give a personalized answer that respects those facts; do not give generic advice when the memory has a specific preference.
6. Distinguish user facts and completed actions from assistant examples, generic advice, hypotheticals, and unsupported plans. Use assistant text only when the question asks about assistant information or the assistant is explicitly restating a user-specific fact.
7. If evidence mentions a related but mismatched entity, place, title, person, or slot, do not answer the mismatched fact.

Final answer rules:
- Answer the exact slot asked by the question. If the question asks where, give the place; if it asks when, give the time/date; if it asks how many, give the number.
- Include the complete final count/list/total/order when the question asks for one.
- If the correct answer is a named entity with an extra location or descriptor, include the main named entity at minimum.
- If the requested information is genuinely missing or only a mismatched fact is recalled, answer exactly: The information is not available in the recalled memory.
- Put the final answer after "ANSWER:".

{evidence_text}

Question: {question}

Think briefly using the ledger, then give the final answer after "ANSWER:"."""


GPT41_KUMSTR_LEDGER_ANSWER_PROMPT = """You answer LongMemEval memory questions using only recalled memory evidence.

Question date: {question_date}
Use the question date for relative words in the question. Use each memory row's own date for relative words inside that memory row.
Use only the recalled memories. Do not use outside knowledge.

Read all recalled memory rows before deciding. The answer may be in a later row or may require combining rows from different sessions.

Before answering, make a compact internal ledger:
- answer_slot: what exact slot the question asks for, such as who, what, where, when, how many, how much, latest/current value, order, or comparison.
- candidate_rows: every row that could affect that slot.
- include: user-specific facts, completed actions, explicit values, and assistant text that restates a user-specific fact.
- exclude: generic assistant advice, examples, hypotheticals, unrelated entities, and mismatched slots.

Task rules:
- For counts, lists, sums, totals, or comparisons, enumerate all included candidates from all rows, deduplicate repeated mentions of the same real-world item/event/value, then compute.
- For temporal/order questions, resolve relative dates against the row date, sort dated facts, and answer the requested order or interval.
- For latest/current/updated facts, the latest relevant user-stated value wins. A later assistant statement saying it lacks memory does not invalidate a user-stated update in that same row.
- For direct fact questions, answer the exact slot asked by the question. Do not answer a date when the question asks for a place, or a related item when it asks for a different entity.
- If evidence is insufficient after scanning all rows, answer exactly: The information is not available in the recalled memory.

Final answer rules:
- Keep the final answer concise but complete.
- For lists/counts/totals/orders, include the complete final result.
- Put the final answer after "ANSWER:".

{evidence_text}

Question: {question}

Work through the ledger, then give the final answer after "ANSWER:"."""


def get_calibrated_longmemeval_prompt(task: str, question: str, answer: str, response: str, abstention: bool = False) -> str:
    prompt = get_longmemeval_prompt(task, question, answer, response, abstention=abstention)
    if abstention:
        return prompt
    if task in {"single-session-user", "single-session-assistant", "multi-session"}:
        prompt = prompt.replace(
            "If the response only contains a subset of the information required by the answer, answer no.",
            "If the correct answer is a named entity with an extra location or descriptive qualifier, and the response contains the main named entity unambiguously, treat it as equivalent and answer yes. If the response only contains a subset of required list, count, or multi-part information, answer no.",
        )
    return prompt


def get_gpt41_calibrated_longmemeval_prompt(task: str, question: str, answer: str, response: str, abstention: bool = False) -> str:
    if abstention:
        return (
            "You are judging a LongMemEval response. Return yes or no only.\n\n"
            "This question is unanswerable from memory. Answer yes if the model clearly says the information is unavailable, incomplete, not mentioned, or explains that only a mismatched related fact was found. Answer no if the model gives a concrete answer to the unanswerable question.\n\n"
            f"Question: {question}\n\nExplanation: {answer}\n\nModel Response: {response}\n\nIs the model response correct? Answer yes or no only."
        )
    if task == "single-session-preference":
        return (
            "You are judging a LongMemEval personalized preference answer. Return yes or no only.\n\n"
            "Answer yes if the response uses the user's core personal preference, constraint, platform, style, anti-preference, or prior interest correctly. It does not need to cover every rubric detail. Answer no if it is generic, contradicts the user's preference, uses the wrong platform/category, or misses the core personalization.\n\n"
            f"Question: {question}\n\nRubric: {answer}\n\nModel Response: {response}\n\nIs the model response correct? Answer yes or no only."
        )
    if task == "temporal-reasoning":
        extra = " For elapsed-time answers, allow equivalent units and off-by-one day differences."
    elif task == "knowledge-update":
        extra = " For updated facts, judge the latest required value as correct even if the response also mentions older values."
    else:
        extra = ""
    return (
        "You are judging a LongMemEval answer. Return yes or no only.\n\n"
        "Answer yes if the model response contains the correct answer or a clearly equivalent value. For a named entity with extra location or descriptive qualifiers, the main named entity is enough if it is unambiguous. For direct slot questions, the response must answer the requested slot; for example, a date alone is not correct for a where question. For counts, lists, sums, comparisons, or multi-part answers, a subset is not enough. Do not require exact wording when the meaning is the same."
        f"{extra}\n\n"
        f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\nIs the model response correct? Answer yes or no only."
    )


def parse_args() -> argparse.Namespace:
    load_project_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/longmemeval/longmemeval_s_dev100_seed20260520.json")
    parser.add_argument("--source-run", default="runs/lme_s_dev100_06b_oldbaseline_rerun_20260521")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--answer-model", default=os.getenv("ANSWER_MODEL") or os.getenv("ANSWERER_MODEL") or os.getenv("LLM_MODEL"))
    parser.add_argument("--answer-base-url", default=os.getenv("ANSWER_BASE_URL") or os.getenv("ANSWERER_BASE_URL") or os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL"))
    parser.add_argument("--judge-base-url", default=os.getenv("JUDGE_BASE_URL") or os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--mode", choices=["answer", "judge-existing"], default="answer")
    parser.add_argument(
        "--answer-style",
        choices=[
            "calibrated_short",
            "mem0_lme",
            "truememory_lme",
            "global_reasoning",
            "typed_route_mem0_kumstr",
            "structured_reasoning",
            "structured_route_kumstr",
            "ms_evidence_table",
            "gpt41_ledger",
            "gpt41_route_ledger_kumstr",
        ],
        default="calibrated_short",
    )
    parser.add_argument("--judge-style", choices=["official", "calibrated", "gpt41_calibrated"], default="official")
    parser.add_argument("--question-ids")
    parser.add_argument("--question-types")
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.judge_model:
        raise RuntimeError("JUDGE_MODEL is required.")
    run_dir = ensure_dir(Path(args.output_dir) / args.run_id)
    source_run = Path(args.source_run)
    samples = load_samples(
        args.data,
        question_ids=load_question_ids(args.question_ids),
        question_types=parse_question_types(args.question_types),
    )
    sample_by_id = {sample.question_id: sample for sample in samples}
    retrieval_by_id = {row["question_id"]: row for row in read_jsonl(source_run / "retrieval_results.jsonl")}
    write_json(run_dir / "config.json", {**vars(args), "num_selected_samples": len(samples)})

    if args.mode == "judge-existing":
        write_predictions(run_dir / "predictions.jsonl", [{"question_id": qid, "hypothesis": value} for qid, value in read_predictions(source_run / "predictions.jsonl").items()])
        if (source_run / "answer_logs.jsonl").exists():
            write_jsonl(run_dir / "answer_logs.jsonl", read_jsonl(source_run / "answer_logs.jsonl"))
    else:
        answer_logs = run_answer_stage(args, run_dir, samples, retrieval_by_id)
        write_token_stats(run_dir, answer_logs, [])

    judge_logs = run_judge_stage(args, run_dir, samples)
    answer_logs = read_jsonl(run_dir / "answer_logs.jsonl")
    write_token_stats(run_dir, answer_logs, judge_logs)
    write_metrics(run_dir, samples, retrieval_by_id, judge_logs)


def run_answer_stage(args: argparse.Namespace, run_dir: Path, samples: list[Any], retrieval_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if not args.answer_model:
        raise RuntimeError("ANSWER_MODEL or ANSWERER_MODEL is required.")
    pred_path = run_dir / "predictions.jsonl"
    log_path = run_dir / "answer_logs.jsonl"
    existing_predictions = read_predictions(pred_path) if args.resume and pred_path.exists() else {}
    existing_logs = {
        row["question_id"]: row
        for row in (read_jsonl(log_path) if args.resume and log_path.exists() else [])
        if "question_id" in row
    }
    if not args.resume:
        pred_path.unlink(missing_ok=True)
        log_path.unlink(missing_ok=True)

    def answer_one(sample: Any) -> tuple[dict[str, str], dict[str, Any]]:
        retrieval = retrieval_by_id[sample.question_id]
        if args.answer_style == "ms_evidence_table" and sample.question_type == "multi-session":
            evidence = format_ms_evidence_table(retrieval["deduped_evidence"], sample.question_date)
        else:
            evidence = format_evidence_for_answerer(
                retrieval["deduped_evidence"],
                sample.question_date,
                question_type=sample.question_type,
            )
        messages = build_answer_messages(args.answer_style, sample.question_date, evidence.text, sample.question, sample.question_type)
        started = time.time()
        body = chat(args.answer_model, args.answer_base_url, os.getenv("ANSWER_API_KEY") or os.getenv("ANSWERER_API_KEY") or os.getenv("OPENAI_API_KEY"), messages, f"answer_prompt_{args.answer_style}")
        latency = time.time() - started
        raw_hypothesis = body["choices"][0].get("message", {}).get("content", "").strip()
        hypothesis = final_answer(args.answer_style, raw_hypothesis)
        usage = dict(body.get("usage", {}))
        prompt_tokens = int(usage.get("prompt_tokens") or estimate_messages_tokens(messages))
        completion_tokens = int(usage.get("completion_tokens") or estimate_tokens(raw_hypothesis))
        log = {
            "question_id": sample.question_id,
            "question_type": sample.question_type,
            "model": args.answer_model,
            "latency_seconds": latency,
            "query_input_tokens": prompt_tokens,
            "query_output_tokens": completion_tokens,
            "query_total_tokens": prompt_tokens + completion_tokens,
            "answer_prompt_style": args.answer_style,
            "raw_hypothesis": raw_hypothesis,
            "answer_input_tokens": prompt_tokens,
            "answer_output_tokens": completion_tokens,
            "answer_total_tokens": prompt_tokens + completion_tokens,
            "provider_usage": usage,
            "evidence_ids": evidence.included_turn_ids,
            "top_k": retrieval.get("top_k"),
            "message_range": retrieval.get("message_range"),
            "retrieval_method": retrieval.get("retrieval_method"),
            "temporal_boost": retrieval.get("temporal_boost"),
            "evidence_truncated": evidence.truncated,
        }
        return {"question_id": sample.question_id, "hypothesis": hypothesis}, log

    pending_samples = [sample for sample in samples if sample.question_id not in existing_predictions]
    predictions = [{"question_id": question_id, "hypothesis": hypothesis} for question_id, hypothesis in existing_predictions.items()]
    logs = list(existing_logs.values())
    with ThreadPoolExecutor(max_workers=args.parallelism) as executor:
        futures = [executor.submit(answer_one, sample) for sample in pending_samples]
        for future in as_completed(futures):
            prediction, log = future.result()
            predictions.append(prediction)
            logs.append(log)
            append_jsonl(pred_path, prediction)
            append_jsonl(log_path, log)
    return logs


def build_answer_messages(answer_style: str, question_date: str, evidence_text: str, question: str, question_type: str) -> list[dict[str, str]]:
    if answer_style == "ms_evidence_table":
        if question_type == "multi-session":
            prompt = MS_EVIDENCE_TABLE_ANSWER_PROMPT.format(question_date=question_date, evidence_text=evidence_text, question=question)
            return [{"role": "user", "content": prompt}]
        return build_baseline_answer_messages(question_date, evidence_text, question, question_type)
    if answer_style == "gpt41_ledger":
        prompt = GPT41_LEDGER_ANSWER_PROMPT.format(question_date=question_date, evidence_text=evidence_text, question=question)
        return [{"role": "user", "content": prompt}]
    if answer_style == "gpt41_route_ledger_kumstr":
        if question_type in {"knowledge-update", "multi-session", "temporal-reasoning"}:
            prompt = GPT41_KUMSTR_LEDGER_ANSWER_PROMPT.format(question_date=question_date, evidence_text=evidence_text, question=question)
            return [{"role": "user", "content": prompt}]
        return build_baseline_answer_messages(question_date, evidence_text, question, question_type)
    if answer_style == "structured_route_kumstr":
        if question_type in {"knowledge-update", "multi-session", "temporal-reasoning"}:
            prompt = STRUCTURED_REASONING_ANSWER_PROMPT.format(question_date=question_date, evidence_text=evidence_text, question=question)
            return [{"role": "user", "content": prompt}]
        return build_baseline_answer_messages(question_date, evidence_text, question, question_type)
    if answer_style == "typed_route_mem0_kumstr":
        if question_type in {"knowledge-update", "multi-session", "temporal-reasoning"}:
            prompt = MEM0_LME_ANSWER_PROMPT.format(question_date=question_date, evidence_text=evidence_text, question=question)
            return [{"role": "user", "content": prompt}]
        return build_baseline_answer_messages(question_date, evidence_text, question, question_type)
    if answer_style == "structured_reasoning":
        prompt = STRUCTURED_REASONING_ANSWER_PROMPT.format(question_date=question_date, evidence_text=evidence_text, question=question)
        return [{"role": "user", "content": prompt}]
    if answer_style == "truememory_lme":
        prompt = TRUEMEMORY_LME_ANSWER_PROMPT.format(question_date=question_date, evidence_text=evidence_text, question=question)
        return [{"role": "user", "content": prompt}]
    if answer_style == "mem0_lme":
        prompt = MEM0_LME_ANSWER_PROMPT.format(question_date=question_date, evidence_text=evidence_text, question=question)
        return [{"role": "user", "content": prompt}]
    if answer_style == "global_reasoning":
        prompt = GLOBAL_REASONING_ANSWER_PROMPT.format(question_date=question_date, evidence_text=evidence_text, question=question)
        return [{"role": "user", "content": prompt}]
    system = ANSWER_SYSTEM_PROMPT.format(question_date=question_date)
    instruction = ANSWER_INSTRUCTIONS.get(question_type, "Answer with the minimal span or phrase that directly satisfies the question.")
    user = (
        f"{evidence_text}\n\n"
        "<QUESTION>\n"
        f"{question}\n"
        "</QUESTION>\n\n"
        "<TASK>\n"
        f"{instruction}\n"
        "Return one concise final answer. Do not include citations, evidence IDs, or reasoning.\n"
        "</TASK>"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def final_answer(answer_style: str, text: str) -> str:
    if answer_style in {"mem0_lme", "truememory_lme", "global_reasoning", "typed_route_mem0_kumstr", "structured_reasoning", "structured_route_kumstr", "ms_evidence_table", "gpt41_ledger", "gpt41_route_ledger_kumstr"} and "ANSWER:" in text:
        return text.rsplit("ANSWER:", 1)[1].strip()
    return text


def format_ms_evidence_table(turns: list[dict[str, Any]], question_date: str) -> Any:
    sorted_turns = sorted(
        turns,
        key=lambda turn: (
            str(turn.get("session_date", "")),
            int(turn.get("session_idx", 0)),
            int(turn.get("turn_idx", 0)),
        ),
    )
    lines = [
        "<MS_EVIDENCE_ROWS>",
        "Rows are recalled historical conversation turns from previous, separate sessions. Treat each row as evidence, not as a final operand.",
        f"Question date: {question_date}",
    ]
    included_turn_ids = []
    for idx, turn in enumerate(sorted_turns, 1):
        included_turn_ids.append(turn["stable_turn_id"])
        lines.extend(
            [
                f"[R{idx:02d}]",
                f"date: {escape_table_cell(turn.get('session_date', ''))}",
                f"session: {escape_table_cell(turn.get('session_id', ''))}",
                f"role: {escape_table_cell(turn.get('role', ''))}",
                f"evidence: {clip_cell(turn.get('content', ''))}",
                "",
            ]
        )
    lines.append("</MS_EVIDENCE_ROWS>")
    text = "\n".join(lines)
    return type(
        "EvidenceTable",
        (),
        {
            "text": text,
            "included_turn_ids": included_turn_ids,
            "truncated": False,
        },
    )()


def escape_table_cell(value: Any) -> str:
    return " ".join(str(value).replace("|", "/").split())


def clip_cell(value: Any, limit: int = 1200) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def parse_question_types(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def run_judge_stage(args: argparse.Namespace, run_dir: Path, samples: list[Any]) -> list[dict[str, Any]]:
    pred_by_id = read_predictions(run_dir / "predictions.jsonl")
    log_path = run_dir / "judge_logs.jsonl"
    existing = {
        row["question_id"]: row
        for row in (read_jsonl(log_path) if args.resume and log_path.exists() else [])
        if "question_id" in row
    }
    if not args.resume:
        log_path.unlink(missing_ok=True)

    def judge_one(sample: Any) -> dict[str, Any]:
        response = pred_by_id[sample.question_id]
        if args.judge_style == "official":
            prompt = get_longmemeval_prompt(sample.question_type, sample.question, sample.answer, response, abstention=is_abstention_sample(sample))
        elif args.judge_style == "calibrated":
            prompt = get_calibrated_longmemeval_prompt(sample.question_type, sample.question, sample.answer, response, abstention=is_abstention_sample(sample))
        else:
            prompt = get_gpt41_calibrated_longmemeval_prompt(sample.question_type, sample.question, sample.answer, response, abstention=is_abstention_sample(sample))
        messages = [{"role": "user", "content": prompt}]
        started = time.time()
        body, content, verdict = chat_judge(args.judge_model, args.judge_base_url, os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY"), messages, f"judge_{args.judge_style}")
        latency = time.time() - started
        usage = dict(body.get("usage", {}))
        prompt_tokens = int(usage.get("prompt_tokens") or estimate_messages_tokens(messages))
        completion_tokens = int(usage.get("completion_tokens") or estimate_tokens(content))
        label = "CORRECT" if verdict == "yes" else "WRONG"
        row = {
            "question_id": sample.question_id,
            "question_type": sample.question_type,
            "model": args.judge_model,
            "judge_style": args.judge_style,
            "label": label,
            "score": 1.0 if label == "CORRECT" else 0.0,
            "raw_response": content,
            "judge_input_tokens": prompt_tokens,
            "judge_output_tokens": completion_tokens,
            "judge_total_tokens": prompt_tokens + completion_tokens,
            "provider_usage": usage,
            "latency_seconds": latency,
        }
        return row

    logs_by_id = dict(existing)
    with ThreadPoolExecutor(max_workers=args.parallelism) as executor:
        futures = [
            executor.submit(judge_one, sample)
            for sample in samples
            if sample.question_id in pred_by_id and sample.question_id not in existing
        ]
        for future in as_completed(futures):
            row = future.result()
            logs_by_id[row["question_id"]] = row
            append_jsonl(log_path, row)
    return [logs_by_id[sample.question_id] for sample in samples if sample.question_id in logs_by_id]


def chat_judge(model: str, base_url: str | None, api_key: str | None, messages: list[dict[str, str]], cache_kind: str) -> tuple[dict[str, Any], str, str]:
    last_content = ""
    for attempt in range(3):
        body = chat(model, base_url, api_key, messages, cache_kind, max_tokens=256, skip_cache=attempt > 0)
        content = judge_response_text(body)
        verdict = parse_yes_no_verdict(content)
        if verdict is not None:
            return body, content, verdict
        last_content = content
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Judge response did not contain yes/no after retries: {last_content!r}")


def judge_response_text(body: dict[str, Any]) -> str:
    message = body["choices"][0].get("message", {})
    content = str(message.get("content") or "").strip()
    if content:
        return content
    return str(message.get("reasoning_content") or "").strip()


def parse_yes_no_verdict(text: str) -> str | None:
    matches = re.findall(r"\b(yes|no)\b", text, flags=re.I)
    if not matches:
        return None
    return matches[-1].lower()


def chat(model: str, base_url: str | None, api_key: str | None, messages: list[dict[str, str]], cache_kind: str, max_tokens: int | None = None, skip_cache: bool = False) -> dict[str, Any]:
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY-compatible key is required.")
    payload_obj: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0}
    if max_tokens is not None:
        payload_obj["max_tokens"] = max_tokens
    if not skip_cache:
        cached = cached_response(cache_kind, model, payload_obj)
        if cached is not None:
            return cached
    payload = json.dumps(payload_obj).encode("utf-8")
    request = urllib.request.Request(
        f"{(base_url or os.getenv('OPENAI_BASE_URL') or 'https://api.openai.com/v1').rstrip('/')}/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    last_error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                body = json.loads(response.read().decode("utf-8"))
            break
        except (urllib.error.URLError, TimeoutError, socket.timeout, http.client.RemoteDisconnected) as exc:
            last_error = exc
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"chat request failed: {last_error}")
    write_cached_response(cache_kind, model, payload_obj, body)
    return body


def write_token_stats(run_dir: Path, answer_logs: list[dict[str, Any]], judge_logs: list[dict[str, Any]]) -> None:
    stats = new_token_summary()
    for row in answer_logs:
        add_query_tokens(stats, row["question_id"], row.get("query_input_tokens", 0), row.get("query_output_tokens", 0))
        add_query_time(stats, row["question_id"], row.get("latency_seconds", 0.0))
    for row in judge_logs:
        add_judge_tokens(stats, row["question_id"], row.get("judge_input_tokens", 0), row.get("judge_output_tokens", 0))
        add_judge_time(stats, row["question_id"], row.get("latency_seconds", 0.0))
    write_json(run_dir / "token_stats.json", stats)


def write_metrics(run_dir: Path, samples: list[Any], retrieval_by_id: dict[str, dict[str, Any]], judge_logs: list[dict[str, Any]]) -> None:
    metrics = qa_metrics(judge_logs)
    write_json(run_dir / "qa_metrics.json", metrics)
    write_json(run_dir / "metrics.json", {"qa": metrics})
    if retrieval_by_id:
        write_jsonl(run_dir / "retrieval_results.jsonl", [retrieval_by_id[sample.question_id] for sample in samples if sample.question_id in retrieval_by_id])
    write_error_analysis(
        samples,
        read_jsonl(run_dir / "retrieval_results.jsonl"),
        run_dir / "predictions.jsonl",
        json.loads((run_dir / "token_stats.json").read_text(encoding="utf-8")),
        run_dir / "error_analysis.jsonl",
        autoeval_log=run_dir / "judge_logs.jsonl",
    )
    print(f"accuracy={metrics['overall_accuracy']:.4f} correct={metrics['correct']} total={metrics['total']} run_dir={run_dir}")


def qa_metrics(logs: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[float]] = defaultdict(list)
    for row in logs:
        by_type[row.get("question_type", "")].append(float(row.get("score", 0.0)))
    correct = sum(1 for row in logs if row.get("label") == "CORRECT")
    total = len(logs)
    return {
        "overall_accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "accuracy_by_question_type": {
            question_type: {
                "accuracy": sum(values) / len(values) if values else 0.0,
                "correct": int(sum(values)),
                "total": len(values),
            }
            for question_type, values in sorted(by_type.items())
        },
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
