#!/usr/bin/env python3
"""Public, editable practice self-test. Never an official grader or score uploader.

Standard library only by default. Optional --verify-tokens uses an already
installed Transformers tokenizer and cached model files; it never downloads.
"""
import argparse
import concurrent.futures
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

VERSION = 'agentist-public-selftest/1.0'
MODEL = 'Qwen/Qwen2.5-7B-Instruct'
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
PROMPT = 'Reply with exactly the uppercase word OK. Do not add explanation or punctuation.'
MAX_BODY = 2 * 1024 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def validate_base(value):
    u = urllib.parse.urlsplit(value)
    if (u.scheme not in ('http', 'https') or u.hostname not in ('127.0.0.1', 'localhost', '::1')
            or u.username or u.password or u.query or u.fragment or u.path not in ('', '/')):
        raise ValueError('Use a loopback base URL (for example http://127.0.0.1:8000); remote access needs an SSH tunnel.')
    return value.rstrip('/')


def request(base, path, key, payload=None, timeout=30):
    headers = {'Accept': 'application/json'}
    if key is not None:
        headers['Authorization'] = 'Bearer ' + key
    data = None if payload is None else json.dumps(payload).encode()
    if data is not None:
        headers['Content-Type'] = 'application/json'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    return opener.open(urllib.request.Request(base + path, data=data, headers=headers), timeout=timeout)


def read_json(response):
    body = response.read(MAX_BODY + 1)
    if len(body) > MAX_BODY:
        raise ValueError('response_too_large')
    return json.loads(body)


def percentile(values, quantile=.95):
    values = sorted(v for v in values if isinstance(v, (float, int)) and math.isfinite(v))
    return values[max(0, math.ceil(len(values) * quantile) - 1)] if values else None


def token_counter(verify, model, revision):
    if not verify:
        return None
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=True, trust_remote_code=False)
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


def stream_once(base, key, model, max_tokens=32, timeout=30, count_tokens=None):
    start = time.perf_counter()
    row = {'ok': False, 'quality_pass': False, 'status': None, 'ttft_ms': None, 'e2e_ms': None,
           'tpot_ms': None, 'event_gap_p95_ms': None, 'content_events': 0,
           'visible_output_tokens': None, 'usage_completion_tokens_unverified': None,
           'token_count_source': 'not_independently_verified', 'error': None}
    times, text, size, done, finished = [], [], 0, False, False
    payload = {'model': model, 'messages': [{'role': 'user', 'content': PROMPT}],
               'temperature': 0, 'max_tokens': max_tokens, 'stream': True,
               'stream_options': {'include_usage': True}}
    try:
        with request(base, '/v1/chat/completions', key, payload, timeout) as response:
            row['status'] = response.status
            if response.status != 200 or 'text/event-stream' not in response.headers.get('Content-Type', ''):
                raise ValueError('not_sse')
            while True:
                if time.perf_counter() - start > timeout:
                    raise TimeoutError()
                line = response.readline(65537)
                if not line:
                    break
                size += len(line)
                if len(line) > 65536 or size > MAX_BODY:
                    raise ValueError('response_too_large')
                if not line.startswith(b'data:'):
                    continue
                data = line[5:].strip()
                if data == b'[DONE]':
                    done = True
                    break
                item = json.loads(data)
                if not isinstance(item, dict) or 'error' in item:
                    raise ValueError('stream_error')
                for choice in item.get('choices', []):
                    if choice.get('index', 0) != 0:
                        raise ValueError('unexpected_choice')
                    reason = choice.get('finish_reason')
                    if reason is not None:
                        if reason not in ('stop', 'length'):
                            raise ValueError('unsuccessful_finish')
                        finished = True
                    content = (choice.get('delta') or {}).get('content')
                    if content is not None and not isinstance(content, str):
                        raise ValueError('invalid_delta')
                    if isinstance(content, str) and content:
                        text.append(content)
                        times.append(time.perf_counter())
                usage = item.get('usage') or {}
                n = usage.get('completion_tokens')
                if type(n) is int and n >= 0:
                    row['usage_completion_tokens_unverified'] = n
        visible = ''.join(text)
        row['ok'] = bool(done and finished and times)
        row['quality_pass'] = row['ok'] and visible.strip() == 'OK'
        row['content_events'] = len(times)
        row['ttft_ms'] = (times[0] - start) * 1000 if times else None
        row['e2e_ms'] = (time.perf_counter() - start) * 1000
        row['event_gap_p95_ms'] = percentile([(b-a)*1000 for a, b in zip(times, times[1:])])
        if count_tokens is not None:
            n = count_tokens(visible)
            if type(n) is not int or n < 0:
                raise ValueError('invalid_token_count')
            row['visible_output_tokens'] = n
            row['token_count_source'] = 'local_tokenizer_visible_text'
            if n > 1 and len(times) > 1:
                row['tpot_ms'] = (times[-1] - times[0]) * 1000 / (n - 1)
        if not row['ok']:
            row['error'] = 'missing_content_finish_or_done'
    except urllib.error.HTTPError as exc:
        row.update(status=exc.code, error='http_' + str(exc.code))
    except (urllib.error.URLError, TimeoutError, OSError):
        row['error'] = 'transport_or_timeout'
    except (ValueError, KeyError, TypeError, AttributeError):
        row['error'] = 'invalid_response'
        row['ok'] = row['quality_pass'] = False
    # Do not echo arbitrary server errors or request/response headers (may contain secrets).
    return row


def check(base, key, model, timeout, count_tokens):
    checks = {}
    for name, path, auth in [('health', '/healthz', None), ('models', '/v1/models', key), ('auth', '/v1/models', None)]:
        try:
            with request(base, path, auth, timeout=timeout) as response:
                body = read_json(response)
                checks[name] = response.status == 200 and (body.get('status') == 'ready' if name == 'health' else
                    any(item.get('id') == model for item in body.get('data', [])) if name == 'models' else False)
        except urllib.error.HTTPError as exc:
            checks[name] = name == 'auth' and exc.code in (401, 403)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, AttributeError, TypeError):
            checks[name] = False
    try:
        with request(base, '/v1/chat/completions', key, {'model': model, 'messages': [{'role': 'user', 'content': PROMPT}],
                     'max_tokens': 32, 'temperature': 0, 'stream': False}, timeout) as response:
            body = read_json(response)
            checks['nonstream'] = response.status == 200 and body['choices'][0]['message']['content'].strip() == 'OK'
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError, AttributeError):
        checks['nonstream'] = False
    row = stream_once(base, key, model, timeout=timeout, count_tokens=count_tokens)
    checks['stream'] = row['ok'] and row['quality_pass']
    return {'checks': checks, 'passed': all(checks.values()), 'stream_sample': row}


def bench(base, key, model, count, concurrency, max_tokens, timeout, count_tokens):
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(lambda _: stream_once(base, key, model, max_tokens, timeout, count_tokens), range(count)))
    seconds = time.perf_counter() - started
    good = [r for r in rows if r['ok'] and r['quality_pass']]
    verified = count_tokens is not None and all(r['visible_output_tokens'] is not None for r in good)
    return {'passed': len(good) == count, 'requests': count, 'concurrency': concurrency,
            'successful': sum(r['ok'] for r in rows), 'correct': len(good), 'window_seconds': seconds,
            'ttft_p95_ms': percentile([r['ttft_ms'] for r in good]),
            'e2e_p95_ms': percentile([r['e2e_ms'] for r in good]),
            'tpot_p95_ms': percentile([r['tpot_ms'] for r in good]),
            'effective_output_tokens_per_second': sum(r['visible_output_tokens'] for r in good)/seconds if verified and good else None,
            'token_count_source': 'local_tokenizer_visible_text' if verified else 'not_independently_verified', 'rows': rows}


def positive_int(low, high):
    def parse(value):
        n = int(value)
        if not low <= n <= high:
            raise argparse.ArgumentTypeError(f'must be between {low} and {high}')
        return n
    return parse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'bench'], help='Explicitly run API checks or a small practice benchmark; --help runs no inference.')
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--api-key-env', default='API_KEY', help='Read key from this environment variable; never pass the secret as an argument.')
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--revision', default=REVISION)
    parser.add_argument('--verify-tokens', action='store_true', help='Use locally installed Transformers + cached tokenizer, never download; requires project venv.')
    parser.add_argument('--requests', type=positive_int(1, 128), default=8)
    parser.add_argument('--concurrency', type=positive_int(1, 16), default=1)
    parser.add_argument('--max-tokens', type=positive_int(1, 256), default=32)
    parser.add_argument('--timeout', type=positive_int(1, 60), default=30)
    parser.add_argument('--out', type=Path, help='Save JSON self-test evidence (no score, no secret).')
    args = parser.parse_args(argv)
    try:
        base = validate_base(args.base_url)
    except ValueError as exc:
        parser.error(str(exc))
    key = os.environ.get(args.api_key_env)
    if not key:
        parser.error('Set the API key environment variable before running tests.')
    try:
        counter = token_counter(args.verify_tokens, args.model, args.revision)
    except Exception:
        parser.error('Local tokenizer unavailable; activate project venv and cache the fixed revision, or omit --verify-tokens (no token rate will be reported).')
    result = check(base, key, args.model, args.timeout, counter) if args.command == 'check' else bench(
        base, key, args.model, args.requests, args.concurrency, args.max_tokens, args.timeout, counter)
    result.update(schema=VERSION, official_score=False, model=args.model, revision=args.revision,
                  notes=['Public editable practice tool; not the official hidden workload, thresholds or score.',
                         'TTFT is first nonempty content event; TPOT is a proxy only with independent token counts.',
                         'SSE event gaps are not exact token ITL. Without --verify-tokens token throughput is null.',
                         'Practice workload asks for OK; do not compare its speed against official structured-task targets.'])
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded + '\n')
    print(encoded)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
