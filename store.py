"""Small SQLite store. Only apply() changes user data."""
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

KINDS = {'task', 'meeting', 'busywork'}
STATUSES = {'open', 'done', 'cancelled'}
FIELDS = {'kind', 'title', 'details', 'scheduled_at', 'status'}
FIELD_LABELS = {'kind': 'Type', 'title': 'Title', 'details': 'Details',
                'scheduled_at': 'Time', 'status': 'Status'}
KIND_LABELS = {'task': 'Task', 'meeting': 'Meeting', 'busywork': 'Administrative work'}
STATUS_LABELS = {'open': 'Open', 'done': 'Done', 'cancelled': 'Cancelled'}


def text(value, name, limit=4000, required=False):
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f'{name} must be text of at most {limit} characters.')
    value = value.strip()
    if required and not value:
        raise ValueError(f'{name} is required.')
    return value


def record_id(value):
    if type(value) is not int or value < 1:
        raise ValueError('record_id must be a positive integer.')
    return value


def checked_fields(fields, creating=False):
    if not isinstance(fields, dict):
        raise ValueError('The record fields must be a JSON object.')
    if not fields:
        raise ValueError('No record fields were supplied.')
    unknown = set(fields) - FIELDS
    if unknown:
        raise ValueError('Unrecognized record field(s): ' + ', '.join(sorted(unknown))
                         + '. Nothing was saved; ask the assistant to correct the action.')
    result = dict(fields)
    for key, value in result.items():
        result[key] = text(value, FIELD_LABELS[key], 200 if key == 'title' else 4000,
                           required=key in {'kind', 'title', 'status'})
    if 'kind' in result and result['kind'] not in KINDS:
        raise ValueError('kind must be task, meeting, or busywork.')
    if 'status' in result and result['status'] not in STATUSES:
        raise ValueError('status must be open, done, or cancelled.')
    if result.get('scheduled_at'):
        try:
            parsed = datetime.fromisoformat(result['scheduled_at'])
            if 'T' not in result['scheduled_at'] or parsed.utcoffset() is None:
                raise ValueError
        except ValueError:
            raise ValueError('Time could not be understood: a valid date, time, and UTC offset are required '
                             '(for example 2026-10-25T15:00+01:00).') from None
    if creating:
        if not {'title', 'kind'} <= result.keys():
            raise ValueError('A new record needs title and kind.')
        result = {'details': '', 'scheduled_at': '', 'status': 'open', **result}
    return result


def display_value(field, value):
    """Format validated values locally; the model never explains its own action."""
    if field == 'kind':
        return KIND_LABELS[value]
    if field == 'status':
        return STATUS_LABELS[value]
    if field == 'scheduled_at':
        if not value:
            return 'Not scheduled'
        date = datetime.fromisoformat(value)
        offset = date.strftime('%z')
        # Keep seconds/fractions and the full offset so the preview stays exact.
        readable_offset = offset[:3] + ':' + offset[3:5]
        if len(offset) > 5:
            readable_offset += ':' + offset[5:]
        return f"{date:%d %B %Y} at {date.time().isoformat()} (UTC{readable_offset})"
    return value if value else '(empty)'


def preview_line(label, value):
    """Indent embedded lines so record text stays distinct from action headings."""
    return label + ': ' + str(value).replace('\n', '\n    ')


@dataclass(frozen=True)
class Proposal:
    """Immutable exact payload plus the version shown to the user."""
    payload: str
    before: str

    def preview(self):
        action = json.loads(self.payload)
        before = json.loads(self.before)
        op = action['op']
        names = {'create': 'ADD', 'update': 'CHANGE', 'delete': 'DELETE', 'add_note': 'ADD NOTE'}
        lines = ["I've generated an action JSON. Here's what it would do :", '',
                 'Action: ' + names[op]]
        if before is not None:
            lines.extend([f"Record ID: {before['id']}", preview_line('Record title', before['title'])])
        if op == 'create':
            for field, label in FIELD_LABELS.items():
                lines.append(preview_line(label, display_value(field, action['fields'][field])))
        elif op == 'update':
            for field, label in FIELD_LABELS.items():
                if field in action['fields']:
                    lines.extend(['', label + ':',
                                  preview_line('  Before', display_value(field, before[field])),
                                  preview_line('  After', display_value(field, action['fields'][field]))])
            lines.append('\nOnly the fields listed above would change.')
        elif op == 'add_note':
            lines.append(preview_line('New note', action['text']))
        else:
            for field, label in FIELD_LABELS.items():
                lines.append(preview_line(label, display_value(field, before[field])))
            lines.append(f"\nWARNING: This permanently deletes this record and its {len(before['notes'])} attached note(s).")
            for note in before['notes']:
                lines.append(preview_line(f"Note {note['id']} to delete", note['body']))
        lines.append('\nNothing has been saved yet. Click I approve to apply this action, or Cancel to reject it.')
        return '\n'.join(lines)


class Store:
    def __init__(self, path):
        if str(path) != ':memory:':
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
            PRAGMA foreign_keys = ON;
            PRAGMA secure_delete = ON;
            PRAGMA temp_store = MEMORY;
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL, title TEXT NOT NULL, details TEXT NOT NULL,
                scheduled_at TEXT NOT NULL, status TEXT NOT NULL,
                revision INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id INTEGER NOT NULL REFERENCES records(id) ON DELETE CASCADE,
                body TEXT NOT NULL
            );
        ''')

    def close(self):
        self.db.close()

    def get(self, ident):
        ident = record_id(ident)
        row = self.db.execute('SELECT * FROM records WHERE id=?', (ident,)).fetchone()
        if row is None:
            raise ValueError(f'Record {ident} does not exist.')
        result = dict(row)
        result['notes'] = [dict(r) for r in self.db.execute(
            'SELECT id, body FROM notes WHERE record_id=? ORDER BY id', (ident,))]
        return result

    def search(self, query='', offset=0):
        query = text(query, 'query', 200)
        if type(offset) is not int or offset < 0:
            raise ValueError('offset must be a nonnegative integer.')
        # instr treats user text literally rather than as SQL or LIKE syntax.
        condition = '''instr(lower(title || ' ' || details || ' ' || scheduled_at), lower(?)) > 0
            OR EXISTS (SELECT 1 FROM notes WHERE notes.record_id=records.id
                       AND instr(lower(body), lower(?)) > 0)'''
        count = self.db.execute('SELECT count(*) FROM records WHERE ' + condition,
                                (query, query)).fetchone()[0]
        rows = self.db.execute('SELECT id, kind, title, scheduled_at, status FROM records WHERE '
                               + condition + ' ORDER BY id LIMIT 20 OFFSET ?',
                               (query, query, offset)).fetchall()
        return {'records': [dict(r) for r in rows], 'total': count,
                'next_offset': offset + 20 if offset + 20 < count else None}

    def propose(self, request):
        if not isinstance(request, dict):
            raise ValueError('An action must be a JSON object.')
        op = request.get('op')
        keys = {'create': {'op', 'fields'}, 'update': {'op', 'record_id', 'fields'},
                'delete': {'op', 'record_id'}, 'add_note': {'op', 'record_id', 'text'}}
        if not isinstance(op, str) or op not in keys:
            raise ValueError('The requested action is unrecognized. Allowed actions: add, change, delete, or add note.')
        unexpected = set(request) - keys[op]
        missing = keys[op] - set(request)
        if unexpected or missing:
            problems = []
            if unexpected:
                problems.append('Unrecognized action field(s): ' + ', '.join(sorted(unexpected)))
            if missing:
                problems.append('Missing required action field(s): ' + ', '.join(sorted(missing)))
            raise ValueError('. '.join(problems) + '.')
        action = dict(request)
        if op in {'create', 'update'}:
            action['fields'] = checked_fields(request['fields'], creating=op == 'create')
        if op == 'add_note':
            action['text'] = text(request['text'], 'note', required=True)
        before = None
        if op != 'create':
            action['record_id'] = record_id(request['record_id'])
            before = self.get(action['record_id'])
        return Proposal(json.dumps(action, sort_keys=True), json.dumps(before, sort_keys=True))

    def apply(self, proposal):
        """Called exclusively by the controller's explicit approval path."""
        action = json.loads(proposal.payload)
        # Revalidate even though ordinary callers only pass prepared proposals.
        checked = self.propose(action)
        if checked != proposal:
            raise ValueError('The record changed. Request a fresh preview.')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            if action['op'] != 'create' and json.dumps(
                    self.get(action['record_id']), sort_keys=True) != proposal.before:
                raise ValueError('The record changed. Request a fresh preview.')
            op = action['op']
            if op == 'create':
                fields = action['fields']
                cursor = self.db.execute('''INSERT INTO records
                    (kind, title, details, scheduled_at, status) VALUES (?, ?, ?, ?, ?)''',
                    tuple(fields[k] for k in ('kind', 'title', 'details', 'scheduled_at', 'status')))
                ident = cursor.lastrowid
            else:
                ident = action['record_id']
                if op == 'delete':
                    self.db.execute('DELETE FROM records WHERE id=?', (ident,))
                else:
                    if op == 'update':
                        fields = action['fields']
                        # Column names come only from the fixed FIELDS whitelist.
                        assignments = ', '.join(f'{key}=?' for key in fields)
                        self.db.execute(f'UPDATE records SET {assignments} WHERE id=?',
                                        (*fields.values(), ident))
                    else:
                        self.db.execute('INSERT INTO notes (record_id, body) VALUES (?, ?)',
                                        (ident, action['text']))
                    self.db.execute('UPDATE records SET revision=revision+1 WHERE id=?', (ident,))
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        return f'Applied {op} for record {ident}.'
