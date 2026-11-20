"""Collect reviewer answers from raw event logs and tally tool use (moderator bookkeeping).

usage: python collect_answers.py <stage> <scratch_dir> <panel_dir>
  - agy stream-json (<scratch>/<stage>_{neon,carbon}_events.jsonl): final text = result.response; tool steps =
    step_update with step_type == "tool" (tool_name listed).
  - codex --json (<scratch>/<stage>_astra_events.jsonl): final text from -o file; tool use = item types other than
    agent_message / reasoning (command_execution, web_search, mcp_tool_call, file_change ...).
Copies raw logs to <panel>/transcripts/ and prints counts.  Does not write the P1_<model>.md files (the moderator
adds a provenance header and writes them).
"""
import collections
import json
import shutil
import sys
from pathlib import Path

stage, scratch, panel = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
names = dict(neon='gemini-4-neon-xhigh', carbon='gemini-4-carbon-high', astra='gpt-6.2-astra')
for short, model in names.items():
    ev = scratch / f'{stage.lower()}_{short}_events.jsonl'
    if not ev.exists():
        print(short, 'missing'); continue
    dst = panel / 'transcripts'
    shutil.copy(ev, dst / f'{stage}_{model}_events.jsonl')
    err = scratch / f'{stage.lower()}_{short}_stderr.txt'
    if err.exists():
        shutil.copy(err, dst / f'{stage}_{model}_stderr.txt')
    lines = [json.loads(l) for l in ev.read_text().splitlines() if l.strip().startswith('{')]
    if short in ('neon', 'carbon'):
        tools = collections.Counter()
        steps = collections.Counter()
        dur = None; text = None; status = None
        for e in lines:
            if e.get('event') == 'step_update':
                s = e['step_update']; steps[s.get('step_type')] += 1
                if s.get('step_type') == 'tool' and s.get('state') in ('ACTIVE', 'DONE'):
                    tools[(s.get('tool_name'), s.get('step_index'))] += 1
            if e.get('event') == 'result':
                r = e['result']; text = r.get('response'); dur = r.get('duration_seconds'); status = r.get('status')
                usage = r.get('usage')
        tool_names = collections.Counter(k[0] for k in tools)
        print(f'{short}: status={status} duration_s={dur} steps={dict(steps)} tool_calls={dict(tool_names)} '
              f'words={len(text.split()) if text else None} usage={usage if text else None}')
        if text:
            (dst / f'{stage}_{model}_response.md').write_text(text)
    else:
        kinds = collections.Counter()
        for e in lines:
            if e.get('type') in ('item.completed', 'item.started'):
                kinds[(e['type'], e['item'].get('type'))] += 1
        final = scratch / f'{stage.lower()}_astra_final.md'
        text = final.read_text() if final.exists() else None
        if final.exists():
            shutil.copy(final, dst / f'{stage}_{model}_final.md')
        usage = [e for e in lines if e.get('type') == 'turn.completed']
        print(f'astra: item kinds={dict(kinds)} words={len(text.split()) if text else None} '
              f'usage={usage[-1].get("usage") if usage else None}')
