import copy
import json
import logging

import pytest

import app as application


VALID_REQUEST = {
    'achievement': 'curry',
    'subject': 'leadership',
    'count': 3,
}

VALID_QUIZ = {
    'achievement_name': 'Curry Achievement',
    'subject': 'Leadership Lab',
    'questions': [
        {
            'q': 'Which statement best describes the CAP Cadet Oath?',
            'choices': ['A. A personal commitment', 'B. A flight plan', 'C. A uniform order', 'D. A mission report'],
            'correct_index': 0,
            'explanation': 'The oath states the cadet commitment and responsibilities.',
            'topic': 'Cadet Oath',
        },
        {
            'q': 'What is the purpose of a chain of command?',
            'choices': ['A. To organize communication', 'B. To assign aircraft', 'C. To record weather', 'D. To replace teamwork'],
            'correct_index': 0,
            'explanation': 'A chain of command provides an orderly path for direction and communication.',
            'topic': 'Chain of Command',
        },
        {
            'q': 'Which action demonstrates customs and courtesies?',
            'choices': ['A. Ignoring rank', 'B. Giving the proper greeting', 'C. Skipping formation', 'D. Changing orders'],
            'correct_index': 1,
            'explanation': 'Proper greetings are a basic example of military customs and courtesies.',
            'topic': 'Customs and Courtesies',
        },
    ],
}


@pytest.fixture
def client():
    application.app.config.update(TESTING=True)
    return application.app.test_client()


def test_home_uses_native_accessible_answer_controls(client):
    response = client.get('/')
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'type="radio"' in html
    assert 'aria-pressed="true"' in html
    assert 'role="progressbar"' in html
    assert 'id="answerFeedback"' in html
    assert 'return `<div class="${cls}" data-i=' not in html
    assert "frame-ancestors 'none'" in response.headers['Content-Security-Policy']


def test_health_and_public_metadata_routes(client):
    assert client.get('/health').get_json() == {'status': 'ok'}
    assert client.get('/robots.txt').status_code == 200
    assert client.get('/sitemap.xml').status_code == 200


@pytest.mark.parametrize(
    'payload',
    [
        None,
        [],
        {},
        {**VALID_REQUEST, 'count': '3'},
        {**VALID_REQUEST, 'count': 2},
        {**VALID_REQUEST, 'count': 16},
        {**VALID_REQUEST, 'count': True},
        {**VALID_REQUEST, 'achievement': 'unknown'},
        {**VALID_REQUEST, 'subject': 'history'},
        {**VALID_REQUEST, 'unexpected': True},
    ],
)
def test_quiz_rejects_invalid_requests(client, payload):
    if payload is None:
        response = client.post('/api/quiz', data='not-json')
    else:
        response = client.post('/api/quiz', json=payload)
    assert response.status_code == 400


def test_quiz_rejects_student_identifier_with_422_before_provider(
    client, monkeypatch, caplog
):
    marker = 'Jane Doe'
    monkeypatch.setattr(
        application,
        '_llm',
        lambda *_args, **_kwargs: pytest.fail('provider must not be called'),
    )
    with caplog.at_level(logging.INFO, logger='capstudy'):
        response = client.post(
            '/api/quiz',
            json={**VALID_REQUEST, 'achievement': f'Student name is {marker}'},
        )
    assert response.status_code == 422
    body = response.get_json()
    assert body['code'] == 'sensitive_data'
    assert 'labeled_name' in body['detected_categories']
    assert marker not in caplog.text
    assert marker not in response.get_data(as_text=True)


def test_quiz_returns_validated_questions_with_private_headers(client, monkeypatch):
    monkeypatch.setattr(
        application,
        '_llm',
        lambda *_args, **_kwargs: '```json\n' + json.dumps(VALID_QUIZ) + '\n```',
    )
    response = client.post('/api/quiz', json=VALID_REQUEST)
    assert response.status_code == 200
    assert response.get_json()['quiz'] == VALID_QUIZ
    assert response.headers['Cache-Control'] == 'private, no-store'
    assert response.headers['X-Robots-Tag'] == 'noindex, nofollow, noarchive'


def test_non_json_provider_output_is_not_logged(client, monkeypatch, caplog):
    marker = 'PRIVATE_QUIZ_OUTPUT_MARKER'
    monkeypatch.setattr(application, '_llm', lambda *_args, **_kwargs: marker)
    with caplog.at_level(logging.WARNING, logger='capstudy'):
        response = client.post('/api/quiz', json=VALID_REQUEST)
    assert response.status_code == 502
    assert marker not in caplog.text
    assert marker not in response.get_data(as_text=True)
    assert 'llm_output_invalid reason=non_json' in caplog.text


@pytest.mark.parametrize('mutation', ['metadata', 'count', 'choices', 'index'])
def test_invalid_quiz_contract_is_rejected(client, monkeypatch, mutation):
    invalid = copy.deepcopy(VALID_QUIZ)
    if mutation == 'metadata':
        invalid['subject'] = 'Aerospace Education'
    elif mutation == 'count':
        invalid['questions'].pop()
    elif mutation == 'choices':
        invalid['questions'][0]['choices'][1] = invalid['questions'][0]['choices'][0]
    else:
        invalid['questions'][0]['correct_index'] = 4
    monkeypatch.setattr(application, '_llm', lambda *_args, **_kwargs: json.dumps(invalid))
    response = client.post('/api/quiz', json=VALID_REQUEST)
    assert response.status_code == 502
    assert 'could not be validated' in response.get_json()['error']


def test_provider_output_with_student_identifier_is_rejected(
    client, monkeypatch, caplog
):
    marker = 'Jane Doe'
    invalid = copy.deepcopy(VALID_QUIZ)
    invalid['questions'][0]['q'] = (
        f'Student name is {marker}; which answer is correct?'
    )
    monkeypatch.setattr(application, '_llm', lambda *_args, **_kwargs: json.dumps(invalid))
    with caplog.at_level(logging.WARNING, logger='capstudy'):
        response = client.post('/api/quiz', json=VALID_REQUEST)
    assert response.status_code == 502
    assert marker not in caplog.text
    assert marker not in response.get_data(as_text=True)


def test_provider_exception_text_and_traceback_are_not_logged(caplog):
    marker = 'PRIVATE_PROVIDER_EXCEPTION_MARKER'
    with caplog.at_level(logging.ERROR, logger='freshsky_common.llm'):
        try:
            raise RuntimeError(marker)
        except RuntimeError:
            logging.getLogger('freshsky_common.llm').exception(
                'LLM provider leaked %s', marker
            )
    assert marker not in caplog.text
    assert 'llm_provider_exception' in caplog.text


def test_metrics_and_privacy_are_private_and_current(client):
    metrics = client.get('/metrics')
    assert metrics.headers['Cache-Control'] == 'private, no-store'
    assert metrics.headers['X-Robots-Tag'].startswith('noindex')
    privacy = client.get('/privacy').get_data(as_text=True)
    assert 'Last updated 2026-07-16' in privacy
    assert 'never provider output or quiz answers' in privacy
    assert 'Google Gemini' not in privacy
