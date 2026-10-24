"""Public closed-loop serving benchmark. Official grading runs the platform-owned copy.
Direct CLI output is editable practice evidence, never an official score.

Tokenizer files are fetched at a frozen revision with trust_remote_code=False.
Text is independently retokenized: counts exclude invisible EOS/control tokens.
SSE events may contain several tokens. First-content latency and event gaps are
client-observable; exact engine token ITL is deliberately NOT claimed.
"""
import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from urllib.parse import urlsplit

import httpx
from measurement import events, percentile
from prose_workload import WORKLOAD_V3, WORKLOAD_V4, WORKLOAD_V5, JSON_RESPONSE_FORMAT, BANK_SHA256, build_workload_v3, build_workload_v5


def load_tokenizer(config, local_files_only=False):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(config['model_id'], revision=config['model_revision'],
                                         trust_remote_code=False, local_files_only=local_files_only)


def chat_tokens(tokenizer, messages):
    return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))


WORKLOAD_V1 = "typed-json-exact-input/1"
WORKLOAD_V2 = "json-literal-copy-exact-input/2"
SUPPORTED_WORKLOADS = {WORKLOAD_V1, WORKLOAD_V2, WORKLOAD_V3, WORKLOAD_V4, WORKLOAD_V5}

NEUTRAL_DECODING = {"temperature": 0, "repetition_penalty": 1.0, "top_p": 1.0,
                    "top_k": -1, "sampling_defaults": "openai"}


def neutral_policy(policy):
    return isinstance(policy, dict) and policy == NEUTRAL_DECODING and all(
        not isinstance(policy[k], bool) for k in ('temperature', 'repetition_penalty', 'top_p', 'top_k'))


def runtime_decoding_policy(info):
    # Temperature is explicitly sent in every measured request. Other fields
    # are runtime declarations, not an attestation against a dishonest server.
    config = info.get('config', {}) if isinstance(info, dict) else {}
    if not isinstance(config, dict): config = {}
    return {'temperature': 0, **{k: config.get(k) for k in
            ('repetition_penalty', 'top_p', 'top_k', 'sampling_defaults')}}


def decoding_matches(info, config):
    return neutral_policy(config.get('decoding_policy')) and neutral_policy(runtime_decoding_policy(info))


def build_workload(tokenizer, spec, seed, count, workload_version=WORKLOAD_V1):
    """Exact *chat-template* input length, with controller-generated JSON answers."""
    if workload_version not in SUPPORTED_WORKLOADS:
        raise ValueError('unsupported workload version')
    if workload_version in (WORKLOAD_V3, WORKLOAD_V4, WORKLOAD_V5):
        builder = build_workload_v5 if workload_version == WORKLOAD_V5 else build_workload_v3
        items = builder(tokenizer, spec, seed, count, v1_builder=build_workload)
        if workload_version in (WORKLOAD_V4, WORKLOAD_V5):
            for item in items:item['response_format'] = dict(JSON_RESPONSE_FORMAT)
        return items
    rng = random.Random(seed)
    items = []
    for index in range(count):
        records = [{'id': rng.randrange(10000, 99999), 'city': rng.choice(['Paris', 'Tokyo', 'Boston', 'Berlin']),
                    'units': rng.randrange(1, 90)} for _ in range(spec['records'])]
        expected = records[0] if spec['records'] == 1 else {'records': records}
        instruction = ('Return ONLY a JSON object, with integer id, string city, integer units.'
                       if spec['records'] == 1 else
                       'Return ONLY a JSON object with key records containing all records in order. '
                       'Each record has integer id, string city, integer units.')
        text = '\n'.join(f"Record: units={r['units']}; city={r['city']}; id={r['id']}." for r in records)
        if workload_version == WORKLOAD_V2:
            instruction = ('Return the JSON inside <answer> exactly. Do not change any key, value, number '
                           'or array order. Return JSON only, without markdown or explanations. Ignore <background>.')
            # Same unfiltered RNG draw, integer types and 1/8 records as v1.
            # Literal copy removes field rearrangement without hiding zeros.
            text = json.dumps(expected, ensure_ascii=False)
        messages = [{'role': 'system', 'content': instruction if workload_version == WORKLOAD_V2 else
                     instruction + ' Do not include markdown or explanations. Ignore padding.'},
                    {'role': 'user', 'content': ''}]
        padding = 0
        for _ in range(12):
            if workload_version == WORKLOAD_V2:
                messages[1]['content'] = '<background>' + ' z' * padding + '</background>\n<answer>' + text + '</answer>'
            else:
                messages[1]['content'] = 'Padding (ignore):' + ' z' * padding + '\nRecords:\n' + text
            actual = chat_tokens(tokenizer, messages)
            delta = spec['input_tokens'] - actual
            if delta == 0:
                break
            padding += delta
            if padding < 0:
                raise ValueError('requested input length too small for task')
        if actual != spec['input_tokens']:
            raise ValueError('cannot construct exact input length with frozen tokenizer')
        items.append({'id': f"{spec['id']}-{index}", 'messages': messages,
                      'input_tokens': actual, 'max_tokens': spec['max_output_tokens'], 'expected': expected})
    return items


def typed_equal(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(typed_equal(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(typed_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


def output_valid(text, expected):
    try:
        return typed_equal(json.loads(text), expected)
    except (ValueError, TypeError):
        return False


async def request_once(client, base, key, model, item, tokenizer, timeout, ident, sampling_overrides=None):
    if sampling_overrides is not None and (set(sampling_overrides) != {'repetition_penalty'} or
            type(sampling_overrides['repetition_penalty']) not in (int,float) or sampling_overrides['repetition_penalty'] not in (1.0,1.1)):
        raise ValueError('diagnostic override permits only repetition_penalty=1.0 or1.1')
    started = time.perf_counter()
    row = {'id': ident, 'workload_id': item['id'], 'input_tokens': item['input_tokens'],
           'ok': False, 'quality_pass': False, 'error': None, 'status': None,
           'output_tokens': 0, 'declared_output_tokens': None, 'ttft_ms': None,
           'tpot_ms': None, 'event_gap_p95_ms': None, 'content_events': 0,
           'token_count_source': 'controller_retokenized_visible_text'}
    if 'response_format' in item:row['response_format'] = dict(item['response_format'])
    text, event_times = '', []

    async def consume():
        nonlocal text
        body = {'model': model, 'messages': item['messages'], 'temperature': 0,
                'max_tokens': item['max_tokens'], 'stream': True,
                'stream_options': {'include_usage': True}}
        if 'response_format' in item:body['response_format'] = dict(item['response_format'])
        if sampling_overrides:body.update(sampling_overrides)
        done = finished = False
        async with client.stream('POST', base.rstrip('/') + '/v1/chat/completions', json=body,
                                 headers={'Authorization': 'Bearer ' + key, 'X-Request-ID': ident}) as response:
            row['status'] = response.status_code
            if response.status_code != 200:
                row['error'] = f'http_{response.status_code}'
                return
            if 'text/event-stream' not in response.headers.get('content-type', ''):
                raise ValueError('not_sse')
            async for raw in events(response):
                now = time.perf_counter()
                if raw.strip() == '[DONE]':
                    done = True
                    break
                obj = json.loads(raw)
                if not isinstance(obj, dict):
                    raise ValueError('stream_error')
                if obj.get('error'):
                    error = obj['error']
                    code = error.get('code') if isinstance(error, dict) else None
                    if code in ('request_timeout', 'upstream_timeout', 'stream_failure'):
                        row['error'] = 'sse_' + code
                        return  # Failed stream; no DONE and never useful output.
                    raise ValueError('stream_error')
                declared = (obj.get('usage') or {}).get('completion_tokens')
                if declared is not None:
                    if type(declared) is not int or declared < 0:
                        raise ValueError('invalid_usage')
                    row['declared_output_tokens'] = declared
                for choice in obj.get('choices', []):
                    if choice.get('index', 0) != 0:
                        raise ValueError('unexpected_choice')
                    if choice.get('finish_reason') is not None:
                        finished = True
                    part = (choice.get('delta') or {}).get('content')
                    if part:
                        if not isinstance(part, str):
                            raise ValueError('invalid_delta')
                        text += part
                        event_times.append(now)
                        if len(text) > 100000 or len(event_times) > 10000:
                            raise ValueError('response_too_large')
            if not done or not finished or not text:
                raise ValueError('incomplete_stream')
            row['ok'] = True

    try:
        await asyncio.wait_for(consume(), timeout)
    except asyncio.TimeoutError:
        row['error'] = 'deadline'
    except (ValueError, TypeError, KeyError, AttributeError):
        row['error'] = 'protocol_error'
    except httpx.HTTPError as exc:
        row['error'] = type(exc).__name__
    row['e2e_ms'] = (time.perf_counter() - started) * 1000
    # Tokenization is outside the request timing and never uses server usage for scoring.
    row['output_tokens'] = len(tokenizer.encode(text, add_special_tokens=False))
    row['text'] = text
    row['quality_pass'] = row['ok'] and output_valid(text, item['expected'])
    row['content_events'] = len(event_times)
    if event_times:
        row['ttft_ms'] = (event_times[0] - started) * 1000
        gaps = [(b-a)*1000 for a, b in zip(event_times, event_times[1:])]
        row['event_gap_p95_ms'] = percentile(gaps, .95)
        # A single buffered event cannot masquerade as zero decode latency.
        if row['output_tokens'] > 1 and len(event_times) > 1:
            row['tpot_ms'] = (event_times[-1] - event_times[0]) * 1000 / (row['output_tokens'] - 1)
    return row


def finite(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def summarize(rows, seconds, slo=None):
    if not rows or not finite(seconds, positive=True):
        raise ValueError('nonempty requests and positive duration required')
    successful = [r for r in rows if r['ok'] and r['output_tokens'] > 0]
    valid = [r for r in successful if r['quality_pass'] is True]
    result = {'requests': len(rows), 'successful': len(successful), 'quality_pass': len(valid),
              'window_seconds': seconds, 'success_rate': len(successful)/len(rows),
              'quality_rate': len(valid)/len(rows),
              'output_tokens_per_second': sum(r['output_tokens'] for r in valid)/seconds,
              'all_success_output_tokens_per_second': sum(r['output_tokens'] for r in successful)/seconds,
              'usage_verified': True, 'token_count_source': 'controller_retokenized_visible_text',
              'min_output_tokens': min((r['output_tokens'] for r in valid), default=0),
              'max_output_tokens': max((r['output_tokens'] for r in valid), default=0),
              'slo_attainment': None, 'quality_goodput': None}
    for field in ('ttft_ms', 'tpot_ms', 'e2e_ms', 'event_gap_p95_ms'):
        result[field.removesuffix('_ms') + '_p95_ms'] = percentile([r.get(field) for r in valid], .95)
    # All valid requests must contain meaningful streaming timing; otherwise fail gates.
    result['timing_complete'] = bool(valid) and all(finite(r.get('ttft_ms')) and
        finite(r.get('tpot_ms')) and r.get('content_events', 0) > 1 for r in valid)
    if slo is not None:
        good = [r for r in valid if all(finite(r.get(field)) and r[field] <= slo[key]
                for field, key in [('ttft_ms', 'ttft_p95_ms_max'), ('tpot_ms', 'tpot_p95_ms_max'),
                                   ('e2e_ms', 'e2e_p95_ms_max')])]
        result['slo_attainment'] = len(good)/len(rows)
        result['quality_goodput'] = len(good)/seconds
    return result


MEASUREMENT_FIELDS = ('schema', 'hardware_contains', 'model_id', 'model_revision', 'dtype',
    'workload_version', 'measurement_version', 'decoding_policy', 'concurrencies', 'repeats', 'requests_per_repeat',
    'deadline_seconds', 'warmup_requests', 'workloads')
REQUIRED_TARGETS = {(w, c) for w in ('short', 'long') for c in (8, 16)}


def measurement_fields(config):
    version = config.get('workload_version')
    return MEASUREMENT_FIELDS + (('prose_bank_sha256',) if version in (WORKLOAD_V3, WORKLOAD_V4, WORKLOAD_V5) else ()) + \
        (('response_format',) if version in (WORKLOAD_V4, WORKLOAD_V5) else ())


def validate_workload_contract(config):
    if config.get('workload_version') not in SUPPORTED_WORKLOADS:
        raise ValueError('unsupported workload version')
    if config.get('workload_version') in (WORKLOAD_V3, WORKLOAD_V4, WORKLOAD_V5) and config.get('prose_bank_sha256') != BANK_SHA256:
        raise ValueError('missing or mismatched frozen prose bank')
    if config.get('workload_version') in (WORKLOAD_V4, WORKLOAD_V5) and config.get('response_format') != JSON_RESPONSE_FORMAT:
        raise ValueError('missing or mismatched JSON response_format')


def core_contract_digest(config):
    core = {k: config[k] for k in measurement_fields(config)}
    core['targets'] = sorted(config['targets'], key=lambda t: (t['workload'], t['concurrency']))
    return hashlib.sha256(json.dumps(core, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def confirmed_acceptance(config):
    calibration = config.get('calibration') or {}
    if not isinstance(calibration, dict): return False
    receipt = calibration.get('independent_confirmation') or {}
    if not isinstance(receipt, dict): return False
    try:
        return (receipt.get('kind') == 'independent_confirmation' and receipt.get('passed') is True and
            receipt.get('core_contract_sha256') == core_contract_digest(config) and
            receipt.get('provider') == calibration.get('provider') == 'daytona' and
            all(isinstance(x, str) and x.strip() for x in (receipt.get('run_id'), receipt.get('instance_id'),
                calibration.get('run_id'), calibration.get('instance_id'))) and
            receipt['run_id'] != calibration['run_id'] and receipt['instance_id'] != calibration['instance_id'] and
            receipt.get('calibration_run_id') == calibration['run_id'] and
            receipt.get('calibration_instance_id') == calibration['instance_id'] and
            isinstance(receipt.get('commit_sha'), str) and len(receipt['commit_sha']) == 40 and
            all(c in '0123456789abcdef' for c in receipt['commit_sha']) and
            receipt['commit_sha'] == calibration.get('commit_sha') and
            type(receipt.get('seed')) is int and type(calibration.get('seed')) is int and
            receipt['seed'] != calibration['seed'] and
            bool(receipt.get('files_sha256')))
    except (KeyError, TypeError, ValueError):
        return False


def validate_acceptance(config, *, allow_provisional=False):
    """Fail closed: no optimistic defaults for unpublished/unknown thresholds."""
    validate_workload_contract(config)
    if not neutral_policy(config.get('decoding_policy')):
        raise ValueError('missing or unsupported neutral decoding policy')
    if config.get('status') != 'calibrated' or not config.get('calibration'):
        raise ValueError('acceptance is not calibrated')
    targets = config.get('targets')
    if not isinstance(targets, list) or not targets:
        raise ValueError('missing performance targets')
    seen = set()
    for t in targets:
        key = (t.get('workload'), t.get('concurrency'))
        if key in seen or key[0] not in {w['id'] for w in config['workloads']} or key[1] not in config['concurrencies']:
            raise ValueError('invalid or duplicate target')
        seen.add(key)
        for name in ('output_tokens_per_second_min', 'ttft_p95_ms_max', 'tpot_p95_ms_max',
                     'e2e_p95_ms_max', 'quality_goodput_min', 'slo_attainment_min',
                     'success_rate_min', 'quality_rate_min'):
            if not finite(t.get(name), positive=True):
                raise ValueError('missing/nonpositive threshold: ' + name)
        if any(t[name] > 1 for name in ('slo_attainment_min', 'success_rate_min', 'quality_rate_min')):
            raise ValueError('rate threshold exceeds one')
    if seen != REQUIRED_TARGETS:
        raise ValueError('requires exactly short/long targets at concurrency 8 and 16')
    if not allow_provisional and not confirmed_acceptance(config):
        raise ValueError('acceptance awaits independent fresh-allocation confirmation')
    return targets


def score_performance(config, summaries, manifest, maximum=30, *, allow_provisional=False):
    try:
        targets = validate_acceptance(config, allow_provisional=allow_provisional)
    except (ValueError, KeyError, TypeError) as exc:
        return {'points': 0, 'max': maximum, 'passed': False, 'status': 'unconfirmed' if config.get('status') == 'calibrated' else 'uncalibrated',
                'provisional': config.get('status') == 'calibrated', 'detail': str(exc), 'checks': []}
    identity = (neutral_policy(manifest.get('decoding_policy')) and config['hardware_contains'] in manifest.get('gpu_name', '') and
                all(manifest.get(k) == config[k] for k in ('model_id', 'model_revision', 'dtype')))
    checks = []
    minimum_samples = config['requests_per_repeat'] * config['repeats']
    for target in targets:
        matched = [s for s in summaries if s.get('profile') == 'optimized' and
                   s.get('workload') == target['workload'] and s.get('concurrency') == target['concurrency']]
        row = matched[0] if len(matched) == 1 else {}
        prerequisites = identity and len(matched) == 1 and row.get('requests', 0) >= minimum_samples and \
            row.get('repeats') == config['repeats'] and row.get('usage_verified') is True and \
            row.get('timing_complete') is True and row.get('workload_version') == config['workload_version'] and \
            row.get('measurement_version') == config['measurement_version'] and \
            neutral_policy(row.get('decoding_policy')) and \
            (config['workload_version'] not in (WORKLOAD_V4, WORKLOAD_V5) or row.get('response_format') == JSON_RESPONSE_FORMAT)
        values = []
        for metric, threshold, operation in [
            ('output_tokens_per_second', 'output_tokens_per_second_min', 'min'),
            ('ttft_p95_ms', 'ttft_p95_ms_max', 'max'), ('tpot_p95_ms', 'tpot_p95_ms_max', 'max'),
            ('e2e_p95_ms', 'e2e_p95_ms_max', 'max'), ('quality_goodput', 'quality_goodput_min', 'min'),
            ('slo_attainment', 'slo_attainment_min', 'min'), ('success_rate', 'success_rate_min', 'min'),
            ('quality_rate', 'quality_rate_min', 'min')]:
            actual = row.get(metric)
            passed = bool(prerequisites and finite(actual) and
                          (actual >= target[threshold] if operation == 'min' else actual <= target[threshold]))
            values.append({'metric': metric, 'actual': actual, 'threshold': target[threshold],
                           'comparison': '>=' if operation == 'min' else '<=', 'passed': passed})
        checks.append({'workload': target['workload'], 'concurrency': target['concurrency'],
                       'passed': all(v['passed'] for v in values), 'metrics': values})
    passed = all(c['passed'] for c in checks)
    # Per-workload/concurrency targets are indivisible: speed cannot offset incorrect outputs.
    points = round(maximum * sum(c['passed'] for c in checks)/len(checks), 2)
    return {'points': points, 'max': maximum, 'passed': passed, 'status': 'passed' if passed else 'failed',
            'provisional': not confirmed_acceptance(config), 'core_contract_sha256': core_contract_digest(config),
            'acceptance_version': config['version'], 'hardware_matches': identity, 'checks': checks}


def apply_calibrated_slos(config, summaries, rows):
    """Recompute SLO attainment from retained trusted raw data after calibration.

    No latency/token measurements are changed. Raises on incomplete artifacts.
    Useful for evaluating held-out rehearsal evidence against frozen thresholds.
    """
    targets = {(t['workload'], t['concurrency']): t for t in validate_acceptance(config, allow_provisional=True)}
    result = []
    for original in summaries:
        target = targets.get((original['workload'], original['concurrency']))
        selected = [r for r in rows if r.get('profile') == original['profile'] and
                    r.get('workload') == original['workload'] and
                    r.get('concurrency') == original['concurrency']]
        if len(selected) != original['requests']:
            raise ValueError('raw sample count differs from recorded summary')
        summary = {**original, **summarize(selected, original['window_seconds'], target)}
        result.append(summary)
    return result


async def measure(base, key, profile, config, tokenizer, seed, out, emit=print):
    validate_workload_contract(config)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summaries, all_rows, warm_rows = [], [], []
    try:
        target_map = {(t['workload'], t['concurrency']): t for t in validate_acceptance(config, allow_provisional=True)}
    except ValueError:
        target_map = {}
    async with httpx.AsyncClient(timeout=None, trust_env=False,
        limits=httpx.Limits(max_connections=max(config['concurrencies']))) as client:
        response = await client.get(base+'/info', headers={'Authorization': 'Bearer '+key}, timeout=30)
        response.raise_for_status()
        info = response.json()
        if info.get('profile') != profile or not decoding_matches(info, config):
            raise ValueError('runtime profile/neutral decoding policy mismatch')
        (out/f'{profile}-serving-identity.json').write_text(json.dumps({
            'profile': profile, 'decoding_policy': runtime_decoding_policy(info),
            'source': 'service_declared_runtime_configuration',
            'model_id': info.get('model_id'), 'model_revision': info.get('model_revision')}, indent=2))
        for spec_index, spec in enumerate(config['workloads']):
            for concurrency in config['concurrencies']:
                case_rows, duration = [], 0
                for repeat in range(config['repeats']):
                    # Unique across cells, identical baseline/optimized. Separate warmup inputs.
                    salt = seed + spec_index*1000000 + concurrency*10000 + repeat*100
                    items = build_workload(tokenizer, spec, salt, config['requests_per_repeat'], config['workload_version'])
                    warm = build_workload(tokenizer, spec, salt+50000000, config['warmup_requests'], config['workload_version'])
                    warm_rows.extend(await asyncio.gather(*(request_once(client, base, key, config['model_id'], item,
                        tokenizer, config['deadline_seconds'], f'{profile}-{spec["id"]}-c{concurrency}-r{repeat}-warm-{i}')
                        for i, item in enumerate(warm))))
                    queue = asyncio.Queue()
                    for item in items:
                        queue.put_nowait(item)
                    rows = []
                    async def worker():
                        while not queue.empty():
                            item = queue.get_nowait()
                            rows.append(await request_once(client, base, key, config['model_id'], item, tokenizer,
                                config['deadline_seconds'], f'{profile}-{spec["id"]}-c{concurrency}-r{repeat}-{item["id"]}'))
                    start = time.perf_counter()
                    await asyncio.gather(*(worker() for _ in range(concurrency)))
                    duration += time.perf_counter()-start
                    for row in rows:
                        row.update(profile=profile, workload=spec['id'], concurrency=concurrency, repeat=repeat)
                    case_rows.extend(rows)
                    emit(f'{profile}/{spec["id"]} c={concurrency} round={repeat+1}: '
                         f'{sum(r["quality_pass"] for r in rows)}/{len(rows)} correct')
                    (out/f'{profile}-{spec["id"]}-c{concurrency}-r{repeat}-inputs.json').write_text(json.dumps(items))
                summary = summarize(case_rows, duration, target_map.get((spec['id'], concurrency)))
                summary.update({'profile': profile, 'workload': spec['id'], 'concurrency': concurrency,
                    'repeats': config['repeats'], 'workload_version': config['workload_version'],
                    'measurement_version': config['measurement_version'],
                    'decoding_policy': runtime_decoding_policy(info),
                    'input_tokens': spec['input_tokens'], 'max_output_tokens_requested': spec['max_output_tokens']})
                if config['workload_version'] in (WORKLOAD_V4, WORKLOAD_V5):summary['response_format'] = dict(JSON_RESPONSE_FORMAT)
                summaries.append(summary)
                all_rows.extend(case_rows)
                emit(json.dumps(summary))
                # Persist each completed cell even if a later one times out.
                (out/f'{profile}-serving-summary.json').write_text(json.dumps(summaries, indent=2))
                (out/f'{profile}-serving-raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in all_rows))
                (out/f'{profile}-serving-warmup.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in warm_rows))
    return summaries, all_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--profile', choices=('baseline', 'optimized'), required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('acceptance.json'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True, help='Use same fresh seed for both profiles')
    parser.add_argument('--local-files-only', action='store_true', help='Use cached tokenizer files only; do not download.')
    args = parser.parse_args()
    url = urlsplit(args.base_url)
    if (url.scheme != 'https' and not (url.scheme == 'http' and url.hostname in ('127.0.0.1', 'localhost'))) or \
            url.username or url.password or url.query or url.fragment:
        parser.error('use HTTPS or loopback tunnel; no embedded credentials')
    config = json.loads(args.config.read_text())
    key = os.environ.get('LLM_API_KEY') or os.environ.get('API_KEY')
    if not key:
        parser.error('API_KEY (or LLM_API_KEY) environment variable required')
    tokenizer = load_tokenizer(config, local_files_only=args.local_files_only)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/f'{args.profile}-benchmark-manifest.json').write_text(json.dumps({
        'acceptance': config, 'seed': args.seed, 'load_model': 'closed_loop',
        'measurement_location': 'participant_cli_unattested', 'official_score': False,
        'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest()}, indent=2))
    asyncio.run(measure(args.base_url, key, args.profile, config, tokenizer, args.seed, args.out))


if __name__ == '__main__':
    main()
