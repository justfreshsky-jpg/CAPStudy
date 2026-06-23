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

from flask import Response, Flask, jsonify, render_template, request

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


# Provider calls are centralized in the privacy-restricted shared chain.

from freshsky_common.llm import LLMChain, install_provider_metrics  # noqa: E402

_SHARED_LLM = LLMChain(privacy_profile="us_public")
install_provider_metrics(app)


def _llm_via_shared_chain(system, user):
    return _SHARED_LLM.complete(system=system, user=user) or None


_PROVIDERS = [('shared', _llm_via_shared_chain)]


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


_PRIVACY_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Privacy — CAPStudy</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font-family:system-ui,sans-serif;max-width:760px;margin:40px auto;padding:0 20px;line-height:1.6;color:#0f172a}h1{margin-bottom:.5em}h2{margin-top:1.5em;font-size:1.1rem}a{color:#1e3a8a}</style>
</head><body>
<a href="/">← Back to CAPStudy</a>
<h1>Privacy Policy — CAPStudy</h1>
<p><em>Last updated 2026-05-07</em></p>
<h2>What we collect</h2>
<p>CAPStudy is a stateless tool. We do <strong>not</strong> require accounts. We do <strong>not</strong> store the text or voice input you submit. We do <strong>not</strong> upload member rosters, patient data, or any personally identifying information.</p>
<h2>What we send to AI providers</h2>
<p>The text or voice transcript you submit is sent to one of several US/EU-jurisdiction LLM providers (Groq, Cerebras, Mistral, HuggingFace via Together, Sambanova, Cloudflare Workers AI, or Google Gemini) for processing. None of these providers train on inputs from our paid-tier API calls (Gemini's free tier may; we do not pass PII).</p>
<h2>What gets logged</h2>
<p>Standard request metadata (IP address, timestamp, response code) is logged by Google Cloud Run for operational purposes (debugging, abuse prevention) and rotated automatically per Google retention defaults. We do not associate logs with individual users.</p>
<h2>Cookies</h2>
<p>A Flask session cookie is set to remember ephemeral state during your visit. It expires when you close the browser. No third-party tracking, no advertising cookies.</p>
<h2>Children</h2>
<p>Some of our tools (e.g. CAPStudy) are designed to be used by minors aged 12+. We do not collect any personally identifying information from anyone, including minors. Parents/guardians of cadets aged 12-17 may use the tool freely.</p>
<h2>Contact</h2>
<p>Questions: <a href="https://www.freshskyai.com/contact">Fresh Sky contact page</a>. Operator: Fresh Sky LLC, Somerset County, NJ.</p>
</body></html>"""

_TERMS_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Terms of Use — CAPStudy</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font-family:system-ui,sans-serif;max-width:760px;margin:40px auto;padding:0 20px;line-height:1.6;color:#0f172a}h1{margin-bottom:.5em}h2{margin-top:1.5em;font-size:1.1rem}a{color:#1e3a8a}</style>
</head><body>
<a href="/">← Back to CAPStudy</a>
<h1>Terms of Use — CAPStudy</h1>
<p><em>Last updated 2026-05-07</em></p>
<h2>What this is</h2>
<p>CAPStudy is a free volunteer-built tool offered by Fresh Sky LLC for use by U.S. Civil Air Patrol cadets and senior members. No charge. No contract. No license required.</p>
<h2>What this is not</h2>
<p>CAPStudy is <strong>not</strong> affiliated with any government agency, military service, or official entity. Output is AI-generated and intended as a draft or study aid only — the human user is responsible for verifying accuracy against authoritative current sources before acting on or filing anything.</p>
<h2>Use at your own discretion</h2>
<p>You agree to use the tool in good faith. Do not submit personally identifying information (PII) about third parties, patient health information (PHI), or classified/sensitive operational details. The tool is not designed to handle such data and we do not warrant against any misuse.</p>
<h2>No warranty</h2>
<p>The tool is provided "as is" without warranty of any kind. Fresh Sky LLC disclaims all liability for damages arising from use or misuse of the output.</p>
<h2>Changes</h2>
<p>We may update or discontinue the tool without notice. If a tool is retired, this URL will redirect or be retired in tandem.</p>
<h2>Contact</h2>
<p>Questions: <a href="https://www.freshskyai.com/contact">Fresh Sky contact page</a>.</p>
</body></html>"""


@app.route('/robots.txt')
def _robots():
    return Response(
        "User-agent: *\nAllow: /\nDisallow: /api/\nDisallow: /metrics\nDisallow: /health\n"
        "Sitemap: https://capstudy.freshskyai.com/sitemap.xml\n",
        mimetype='text/plain',
    )


@app.route('/sitemap.xml')
def _sitemap():
    return Response(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        '  <url><loc>https://capstudy.freshskyai.com/</loc><changefreq>weekly</changefreq><priority>1.0</priority></url>\n'
        '</urlset>\n',
        mimetype='application/xml',
    )


@app.route('/privacy')
def _privacy():
    return Response(_PRIVACY_HTML, mimetype='text/html')


@app.route('/terms')
def _terms():
    return Response(_TERMS_HTML, mimetype='text/html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
