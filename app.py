"""
CAPStudy — Civil Air Patrol Achievement Test practice generator.

Cadets take Achievement Tests every 2 months as part of the cadet program.
Each achievement has a Leadership Lab quiz and an Aerospace Education quiz
(alternating cycles). This app generates fresh practice questions on
demand, with explanations, so cadets can study before the real test.

Public-domain content. No PII. No cadet roster uploads. No score tracking
across users (each session is ephemeral).

Built by a CAP member as a free volunteer offering for the cadet program.
"""
import collections
import functools
import json
import logging
import os
import re
import threading

import requests
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(32))
app.config.update(
    SESSION_COOKIE_SECURE=os.environ.get('SESSION_COOKIE_SECURE', 'true').lower() == 'true',
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('capstudy')

_metrics = {'requests_total': 0, 'provider_success': collections.Counter(), 'provider_failure': collections.Counter()}
_metrics_lock = threading.Lock()


def _route_handler(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception:
            logger.exception('Unhandled exception in %s', f.__name__)
            return jsonify(error='An error occurred. Please try again.'), 500
    return wrapper


@app.after_request
def _security_headers(resp):
    resp.headers.setdefault('X-Content-Type-Options', 'nosniff')
    resp.headers.setdefault('X-Frame-Options', 'DENY')
    resp.headers.setdefault('Referrer-Policy', 'strict-origin-when-cross-origin')
    resp.headers.setdefault('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')
    return resp


_HTTP_TIMEOUT = 35


def _llm_via_groq(system, user):
    key = os.environ.get('GROQ_KEY', '')
    if not key: return None
    r = requests.post('https://api.groq.com/openai/v1/chat/completions',
        headers={'Authorization': f'Bearer {key}'},
        json={'model': os.environ.get('GROQ_MODEL', 'llama-3.3-70b-versatile'),
              'messages': [{'role':'system','content':system}, {'role':'user','content':user}],
              'temperature': 0.6, 'response_format': {'type': 'json_object'}},
        timeout=_HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']


def _llm_via_cerebras(system, user):
    key = os.environ.get('CEREBRAS_KEY', '')
    if not key: return None
    r = requests.post('https://api.cerebras.ai/v1/chat/completions',
        headers={'Authorization': f'Bearer {key}'},
        json={'model': os.environ.get('CEREBRAS_MODEL', 'llama-3.3-70b'),
              'messages': [{'role':'system','content':system}, {'role':'user','content':user}],
              'temperature': 0.6},
        timeout=_HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']


def _llm_via_gemini(system, user):
    key = os.environ.get('GEMINI_KEY', '')
    if not key: return None
    r = requests.post(f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={key}',
        headers={'Content-Type':'application/json'},
        json={'system_instruction':{'parts':[{'text':system}]},
              'contents':[{'role':'user','parts':[{'text':user}]}],
              'generationConfig':{'temperature':0.6, 'responseMimeType':'application/json'}},
        timeout=_HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()['candidates'][0]['content']['parts'][0]['text']


def _llm_via_mistral(system, user):
    key = os.environ.get('MISTRAL_KEY', '')
    if not key: return None
    r = requests.post('https://api.mistral.ai/v1/chat/completions',
        headers={'Authorization': f'Bearer {key}'},
        json={'model': os.environ.get('MISTRAL_MODEL', 'mistral-small-latest'),
              'messages': [{'role':'system','content':system}, {'role':'user','content':user}],
              'temperature': 0.6, 'response_format': {'type': 'json_object'}},
        timeout=_HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']


def _llm_via_huggingface(system, user):
    key = os.environ.get('HF_KEY', '')
    if not key: return None
    r = requests.post('https://router.huggingface.co/v1/chat/completions',
        headers={'Authorization': f'Bearer {key}'},
        json={'model': os.environ.get('HF_MODEL', 'meta-llama/Llama-3.3-70B-Instruct'),
              'messages': [{'role':'system','content':system}, {'role':'user','content':user}],
              'temperature': 0.6},
        timeout=_HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']


_PROVIDERS = [
    ('groq', _llm_via_groq),
    ('cerebras', _llm_via_cerebras),
    ('gemini', _llm_via_gemini),
    ('mistral', _llm_via_mistral),
    ('huggingface', _llm_via_huggingface),
]


def _llm(system: str, user: str) -> str:
    last_err = None
    for name, fn in _PROVIDERS:
        try:
            out = fn(system, user)
            if out:
                with _metrics_lock:
                    _metrics['provider_success'][name] += 1
                return out.strip()
        except Exception as e:
            last_err = e
            with _metrics_lock:
                _metrics['provider_failure'][name] += 1
            logger.warning('Provider %s failed: %s', name, e)
    raise RuntimeError(f'All LLM providers failed: {last_err}')


# Cadet achievement progression. Each achievement has Leadership topics and
# Aerospace Education topics; the LLM generates appropriate-difficulty
# questions based on the achievement level.
ACHIEVEMENTS = [
    {'id': 'curry',         'rank': 'C/Amn',     'phase': 1, 'name': 'Curry Achievement',
     'topics_lead': 'CAP basics, oath, motto, mission, customs and courtesies, basic drill positions, chain of command structure',
     'topics_ae':   'introduction to aerospace, why we have an Air Force, what CAP does in aerospace education'},
    {'id': 'arnold',        'rank': 'C/A1C',     'phase': 1, 'name': 'Arnold Achievement',
     'topics_lead': 'CAP organization, follower-to-leader transition, basic drill (facing movements, marching), uniform standards',
     'topics_ae':   'principles of flight basics, four forces, parts of an airplane, types of aircraft'},
    {'id': 'feik',          'rank': 'C/SrA',     'phase': 1, 'name': 'Feik Achievement',
     'topics_lead': 'communication basics, intermediate drill, leadership styles introduction',
     'topics_ae':   'history of flight (Wright Brothers, early aviation), aviation pioneers'},
    {'id': 'wright',        'rank': 'C/SSgt',    'phase': 1, 'name': 'Wright Brothers Award (milestone)',
     'topics_lead': 'phase 1 review, NCO responsibilities introduction, character / integrity',
     'topics_ae':   'early aviation development, military aviation history WWI'},
    {'id': 'lindbergh',     'rank': 'C/TSgt',    'phase': 2, 'name': 'Lindbergh Achievement',
     'topics_lead': 'taking on team responsibilities, leading small groups, mentoring lower cadets',
     'topics_ae':   'navigation principles, pilotage and dead reckoning, charts and headings'},
    {'id': 'rickenbacker',  'rank': 'C/MSgt',    'phase': 2, 'name': 'Rickenbacker Achievement',
     'topics_lead': 'leading by example, character forum facilitation, conflict resolution',
     'topics_ae':   'aviation in WWII, propulsion (piston engines vs. jet engines), aircraft structures'},
    {'id': 'doolittle',     'rank': 'C/SMSgt',   'phase': 2, 'name': 'Doolittle Achievement',
     'topics_lead': 'leading element-sized teams, training plans, drug and substance abuse prevention (DDR)',
     'topics_ae':   'jet age, supersonic flight, transonic regime, basic aerodynamics review'},
    {'id': 'mitchell',      'rank': 'C/2dLt',    'phase': 2, 'name': 'Mitchell Award (milestone)',
     'topics_lead': 'NCO-to-officer transition, decision-making frameworks, ethical leadership, eligibility for cadet officer roles',
     'topics_ae':   'phase 2 AE review, intro to space (orbits, escape velocity)'},
    {'id': 'garber',        'rank': 'C/1stLt',   'phase': 3, 'name': 'Garber Achievement',
     'topics_lead': 'leading larger groups (flights), professional development, mentoring NCOs',
     'topics_ae':   'space exploration milestones (Mercury/Gemini/Apollo), space race'},
    {'id': 'goddard',       'rank': 'C/Capt',    'phase': 3, 'name': 'Goddard Achievement',
     'topics_lead': 'staff officer roles, mission planning, formal counseling techniques',
     'topics_ae':   'rocketry principles, propulsion systems for space, satellite operations'},
    {'id': 'armstrong',     'rank': 'C/Maj',     'phase': 3, 'name': 'Armstrong Achievement',
     'topics_lead': 'leading squadrons, executive presence, formal speeches and briefings',
     'topics_ae':   'human spaceflight, ISS, modern space exploration, commercial spaceflight'},
    {'id': 'earhart',       'rank': 'C/Lt Col',  'phase': 3, 'name': 'Earhart Award (milestone)',
     'topics_lead': 'phase 3 review, cadet officer leadership at squadron and group level',
     'topics_ae':   'phase 3 AE review, aerospace careers and pathways'},
    {'id': 'eaker',         'rank': 'C/Lt Col',  'phase': 4, 'name': 'Eaker Award (milestone)',
     'topics_lead': 'strategic leadership, mentoring across phases, executive-level decision-making',
     'topics_ae':   'aerospace policy, contemporary aerospace issues'},
    {'id': 'spaatz',        'rank': 'C/Col',     'phase': 4, 'name': 'Spaatz Award (highest)',
     'topics_lead': 'capstone leadership, exam-level mastery of all leadership concepts',
     'topics_ae':   'capstone aerospace knowledge, mastery-level synthesis'},
]


def _format_achievements() -> str:
    lines = []
    for a in ACHIEVEMENTS:
        lines.append(f"  {a['id']} ({a['rank']}, Phase {a['phase']}): {a['name']}")
        lines.append(f"    Leadership: {a['topics_lead']}")
        lines.append(f"    Aerospace Ed: {a['topics_ae']}")
    return '\n'.join(lines)


_QUIZ_SYSTEM = (
    "You are a CAP cadet Achievement Test practice question generator. Generate practice questions "
    "appropriate for the given achievement level and subject area (Leadership Lab or Aerospace Education). "
    "Questions should be at the difficulty level of the actual CAP achievement test for that rank.\n\n"
    "Output a JSON object with this structure:\n"
    '{\n'
    '  "achievement_name": "human label",\n'
    '  "subject": "Leadership Lab | Aerospace Education",\n'
    '  "questions": [\n'
    '    {\n'
    '      "q": "the question text",\n'
    '      "choices": ["A. ...", "B. ...", "C. ...", "D. ..."],\n'
    '      "correct_index": 0,\n'
    '      "explanation": "why this is correct, with reference to the topic / concept",\n'
    '      "topic": "short topic tag (e.g., \'Drill\', \'Chain of Command\', \'History of Flight\')"\n'
    '    }\n'
    '  ]\n'
    '}\n\n'
    "RULES:\n"
    "- Output ONLY the JSON object. No prose around it.\n"
    "- Generate exactly the number of questions requested (default 10).\n"
    "- Multiple choice with 4 options (A, B, C, D). correct_index is 0-based.\n"
    "- Difficulty must match the achievement level — Curry questions are basic; Mitchell+ questions involve NCO/officer leadership concepts; Spaatz/Eaker questions are advanced.\n"
    "- Cover a mix of topics from that achievement's curriculum (don't ask 10 questions on the same micro-topic).\n"
    "- Explanations should teach, not just confirm — 1-3 sentences explaining the why behind the answer.\n"
    "- If you genuinely don't know enough about a niche CAP topic to write a fair question, write a question on a more central topic instead.\n"
    "- Stay G-rated. The audience is cadets (12-21 years old).\n\n"
    "ACHIEVEMENT INDEX (use the matching achievement's curriculum):\n"
    + _format_achievements()
)


def _strip_code_fence(s: str) -> str:
    s = s.strip()
    if s.startswith('```'):
        s = re.sub(r'^```[a-zA-Z]*\s*', '', s)
        s = re.sub(r'\s*```\s*$', '', s)
    return s.strip()


@app.route('/')
def index():
    return render_template('index.html', achievements=ACHIEVEMENTS)


@app.route('/health')
def health():
    return jsonify(status='ok')


@app.route('/metrics')
def metrics():
    with _metrics_lock:
        return jsonify({
            'requests_total': _metrics['requests_total'],
            'provider_success': dict(_metrics['provider_success']),
            'provider_failure': dict(_metrics['provider_failure']),
        })


@app.route('/api/quiz', methods=['POST'])
@_route_handler
def quiz():
    data = request.get_json(silent=True) or {}
    achievement_id = (data.get('achievement') or '').strip().lower()
    subject = (data.get('subject') or 'leadership').strip().lower()
    n = max(3, min(15, int(data.get('count') or 10)))
    valid_ids = {a['id'] for a in ACHIEVEMENTS}
    if achievement_id not in valid_ids:
        return jsonify(error='Pick a valid achievement.'), 400
    if subject not in ('leadership', 'aerospace'):
        return jsonify(error='Subject must be leadership or aerospace.'), 400
    subject_label = 'Leadership Lab' if subject == 'leadership' else 'Aerospace Education'

    user_msg = (
        f"Generate {n} multiple-choice practice questions for the **{achievement_id}** achievement, "
        f"subject area: **{subject_label}**. Match the difficulty + curriculum of that achievement level."
    )
    with _metrics_lock:
        _metrics['requests_total'] += 1
    raw = _llm(_QUIZ_SYSTEM, user_msg)
    raw = _strip_code_fence(raw)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning('LLM returned non-JSON: %s', raw[:200])
        return jsonify(error='The model returned an unparseable quiz. Please try again.'), 502
    if 'questions' not in parsed or not isinstance(parsed['questions'], list):
        return jsonify(error='Quiz format invalid. Please try again.'), 502
    return jsonify(quiz=parsed)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
