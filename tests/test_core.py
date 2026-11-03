"""Run with: python -m unittest discover -s tests -v"""
import json
from pathlib import Path
import tempfile
import unittest

from controller import Controller, parse_response
from store import Store


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.store = Store(':memory:')
        self.chat = Controller(self.store)

    def tearDown(self):
        self.store.close()

    def propose_task(self, title='Write report'):
        return self.chat.receive(json.dumps({'op': 'create', 'fields': {
            'title': title, 'kind': 'task'}}))

    def create_task(self):
        self.propose_task()
        self.chat.approve()
        return self.store.get(1)

    def test_proposal_does_not_write_and_typed_approval_cannot_apply(self):
        self.propose_task()
        self.assertEqual(self.store.search()['total'], 0)
        with self.assertRaises(ValueError):
            self.chat.begin('I approve')
        self.assertEqual(self.store.search()['total'], 0)
        self.chat.approve()
        self.assertEqual(self.store.search()['total'], 1)
        with self.assertRaises(ValueError):
            self.chat.approve()
        self.assertEqual(self.store.search()['total'], 1)

    def test_pending_preview_cannot_be_replaced(self):
        self.propose_task('Original')
        with self.assertRaises(ValueError):
            self.propose_task('Replacement')
        self.chat.approve()
        self.assertEqual(self.store.get(1)['title'], 'Original')

    def test_cancel_and_clear_do_not_save(self):
        self.propose_task()
        self.chat.cancel()
        self.propose_task()
        self.chat.clear()
        self.assertEqual(self.store.search()['total'], 0)
        self.assertEqual(self.chat.history, [])
        self.assertIsNone(self.chat.pending)

    def test_notes_updates_and_delete_need_approval(self):
        self.create_task()
        self.chat.receive('{"op":"add_note","record_id":1,"text":"Bring draft"}')
        self.assertEqual(self.store.get(1)['notes'], [])
        self.chat.approve()
        self.assertEqual(self.store.get(1)['notes'][0]['body'], 'Bring draft')
        self.chat.receive('{"op":"update","record_id":1,"fields":{"status":"done"}}')
        self.assertEqual(self.store.get(1)['status'], 'open')
        self.chat.approve()
        self.assertEqual(self.store.get(1)['status'], 'done')
        self.chat.receive('{"op":"delete","record_id":1}')
        self.assertEqual(self.store.search()['total'], 1)
        self.chat.approve()
        self.assertEqual(self.store.search()['total'], 0)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 0)

    def test_stale_preview_is_rejected(self):
        self.create_task()
        self.chat.receive('{"op":"delete","record_id":1}')
        other = self.store.propose({'op': 'add_note', 'record_id': 1, 'text': 'Changed elsewhere'})
        self.store.apply(other)
        with self.assertRaises(ValueError):
            self.chat.approve()
        self.assertEqual(self.store.search()['total'], 1)
        self.assertIsNone(self.chat.pending)

    def test_invalid_or_injected_actions_do_not_write(self):
        invalid = [
            {'op': 'sql', 'query': 'DROP TABLE records'},
            {'op': 'create', 'fields': {'kind': 'task', 'title': 'X', 'approved': True}},
            {'op': 'create', 'fields': {'kind': 'unknown', 'title': 'X'}},
            {'op': 'create', 'fields': {'kind': 'task', 'title': ' '}},
            {'op': 'delete', 'record_id': True},
            {'op': 'create', 'fields': {'kind': 'task', 'title': 'X', 'scheduled_at': 'tomorrow'}},
            {'op': 'create', 'fields': {'kind': 'task', 'title': 'X', 'scheduled_at': '2026-10-25T15:00'}},
            {'op': ['create']},
        ]
        for action in invalid:
            with self.subTest(action=action), self.assertRaises(ValueError):
                self.chat.receive(json.dumps(action))
        self.assertEqual(self.store.search()['total'], 0)

    def test_duplicate_json_fields_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_response('{"op":"reply","op":"delete","record_id":1}')

    def test_read_only_search_and_tool_limit(self):
        self.create_task()
        self.chat.begin('What is saved?')
        for _ in range(5):
            kind, result = self.chat.receive('{"op":"search","query":"report","offset":0}')
            self.assertEqual(kind, 'read')
            self.assertEqual(result['total'], 1)
        with self.assertRaises(ValueError):
            self.chat.receive('{"op":"search","query":"","offset":0}')
        self.assertIsNone(self.chat.pending)

    def test_persistence_contains_approved_content_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'records.sqlite3'
            store = Store(path)
            chat = Controller(store)
            chat.begin('Secret unsaved conversation')
            chat.receive('{"op":"create","fields":{"kind":"task","title":"Saved title"}}')
            chat.approve()
            chat.receive('{"op":"create","fields":{"kind":"task","title":"Unsaved title"}}')
            chat.cancel()
            store.close()
            data = path.read_bytes()
            self.assertIn(b'Saved title', data)
            self.assertNotIn(b'Secret unsaved conversation', data)
            self.assertNotIn(b'Unsaved title', data)
            reopened = Store(path)
            self.assertEqual(reopened.search()['total'], 1)
            reopened.close()

    def test_search_is_literal_and_paginated(self):
        for i in range(23):
            self.store.apply(self.store.propose({'op': 'create', 'fields': {
                'kind': 'task', 'title': f'Task {i}'}}))
        self.assertEqual(len(self.store.search()['records']), 20)
        self.assertEqual(self.store.search()['next_offset'], 20)
        self.assertEqual(len(self.store.search(offset=20)['records']), 3)
        self.assertIsNone(self.store.search(offset=20)['next_offset'])
        self.assertEqual(self.store.search("%' OR 1=1 --")['total'], 0)


if __name__ == '__main__':
    unittest.main()
