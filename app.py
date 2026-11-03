"""Tkinter entry point. Run this file from your CUDA-enabled Python environment."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sqlite3
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

from controller import Controller
from model_worker import ModelRunner
from store import Store
from christ_compendium import SIMULATION_NOTICE, compendium_text


COLORS = {
    'background': '#171a21',
    'panel': '#20252f',
    'button': '#303848',
    'hover': '#414e64',
    'text': '#e6eaf0',
    'muted': '#aeb8c7',
    'accent': '#91c5ff',
    'selection': '#355780',
    'border': '#49566c',
}
TEXT_FONT = ('Segoe UI', 13)
CONTROL_FONT = ('Segoe UI', 12)


def apply_night_theme(root):
    """Use a recolorable ttk theme instead of platform-native light controls."""
    root.configure(background=COLORS['background'])
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure('.', background=COLORS['background'], foreground=COLORS['text'],
                    font=CONTROL_FONT)
    style.configure('TLabel', foreground=COLORS['muted'])
    style.configure('TButton', background=COLORS['button'], padding=(10, 6),
                    bordercolor=COLORS['border'], lightcolor=COLORS['button'],
                    darkcolor=COLORS['button'], focuscolor=COLORS['accent'])
    style.map('TButton', background=[('disabled', COLORS['panel']),
                                   ('pressed', COLORS['selection']), ('active', COLORS['hover'])],
              foreground=[('disabled', COLORS['muted'])])
    style.configure('TCheckbutton', indicatorbackground=COLORS['panel'],
                    indicatorforeground=COLORS['accent'], focuscolor=COLORS['accent'])
    style.map('TCheckbutton', background=[('active', COLORS['background'])],
              indicatorbackground=[('selected', COLORS['selection']), ('active', COLORS['hover'])])
    style.configure('TEntry', fieldbackground=COLORS['panel'],
                    foreground=COLORS['text'], insertcolor=COLORS['text'],
                    selectbackground=COLORS['selection'], selectforeground=COLORS['text'],
                    bordercolor=COLORS['border'])
    style.map('TEntry', bordercolor=[('focus', COLORS['accent'])],
              fieldbackground=[('disabled', COLORS['background'])])
    style.configure('Vertical.TScrollbar', background=COLORS['button'],
                    troughcolor=COLORS['background'], arrowcolor=COLORS['muted'],
                    bordercolor=COLORS['background'], lightcolor=COLORS['button'],
                    darkcolor=COLORS['button'])
    style.map('Vertical.TScrollbar', background=[('active', COLORS['hover']),
                                               ('pressed', COLORS['selection'])])


def text_colors():
    """Shared colors for classic Tk text widgets, which do not use ttk styles."""
    return dict(background=COLORS['panel'], foreground=COLORS['text'],
                insertbackground=COLORS['accent'], selectbackground=COLORS['selection'],
                selectforeground=COLORS['text'], inactiveselectbackground=COLORS['selection'],
                highlightbackground=COLORS['border'], highlightcolor=COLORS['accent'],
                highlightthickness=1, borderwidth=0, padx=10, pady=10,
                relief='flat')


def dark_scrolled_text(master, **options):
    view = scrolledtext.ScrolledText(master, **text_colors(), **options)
    view.frame.configure(background=COLORS['background'])
    # A native Windows Tk scrollbar can ignore colors; use a themed ttk one.
    view.vbar.destroy()
    view.vbar = ttk.Scrollbar(view.frame, orient='vertical', command=view.yview)
    view.vbar.pack(side='right', fill='y', before=view._w)
    view.configure(yscrollcommand=view.vbar.set)
    return view


def default_data_dir():
    base = Path(os.environ.get('LOCALAPPDATA', Path.home() / '.local' / 'share'))
    return base / 'PrivateLocalAssistant'


class Application:
    def __init__(self, root, model_path, data_dir, idle_seconds):
        self.root = root
        apply_night_theme(root)
        self.store = Store(data_dir / 'records.sqlite3')
        self.controller = Controller(self.store)
        self.runner = ModelRunner(model_path, idle_seconds)
        self.approval_frame = None
        self.closed = False
        root.title('Private Local Assistant')
        root.geometry('960x760')
        root.minsize(720, 520)
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.report_callback_exception = self.callback_error
        root.bind_all('<KeyPress>', lambda event: self.runner.touch(), add='+')
        root.bind_all('<ButtonPress>', lambda event: self.runner.touch(), add='+')

        toolbar = ttk.Frame(root, padding=8)
        toolbar.pack(fill='x')
        self.supportive = tk.BooleanVar()
        self.supportive_button = ttk.Checkbutton(toolbar, text='Supportive mode', variable=self.supportive)
        self.supportive_button.pack(side='left')
        self.jesus_mode = tk.BooleanVar()
        self.jesus_button = ttk.Checkbutton(toolbar, text='Jesus mode (simulation)',
                                          variable=self.jesus_mode, command=self.change_jesus_mode)
        self.jesus_button.pack(side='left', padx=8)
        ttk.Button(toolbar, text='Christ quotations', command=self.show_compendium).pack(side='left')
        actions = ttk.Frame(root, padding=(8, 0, 8, 8))
        actions.pack(fill='x')
        ttk.Button(actions, text='Records', command=self.browse).pack(side='left')
        ttk.Button(actions, text='Clear chat', command=self.clear).pack(side='left', padx=8)
        ttk.Button(actions, text='Stop / unload', command=self.unload).pack(side='left')
        self.status = tk.StringVar(value='Model unloaded · offline')
        ttk.Label(root, textvariable=self.status, padding=(8, 0)).pack(anchor='w')
        self.chat = dark_scrolled_text(root, wrap='word', state='disabled', font=TEXT_FONT)
        self.chat.pack(fill='both', expand=True, padx=8, pady=8)
        self.chat.tag_configure('user', foreground=COLORS['accent'])
        self.chat.tag_configure('system', foreground=COLORS['muted'])
        self.chat.tag_configure('warning', foreground='#ffc078')
        composer = ttk.Frame(root, padding=(8, 0, 8, 8))
        composer.pack(fill='x')
        self.input = tk.Text(composer, height=3, wrap='word', font=TEXT_FONT, **text_colors())
        self.input.pack(side='left', fill='x', expand=True)
        self.input.bind('<Control-Return>', self.send)
        self.send_button = ttk.Button(composer, text='Send', command=self.send)
        self.send_button.pack(side='left', padx=(8, 0))
        self.write('System', 'Chat is temporary. Only approved records and notes are saved.\n'
                   'Use Ctrl+Enter to send. Stop / unload also cancels generation.', 'system')
        self.input.focus_set()
        root.after(100, self.poll)

    def write(self, speaker, content, tag=''):
        self.chat.configure(state='normal')
        self.chat.insert('end', f'{speaker}: {content}\n\n', tag)
        self.chat.configure(state='disabled')
        self.chat.see('end')

    def refresh_controls(self):
        blocked = self.runner.busy or self.controller.pending is not None
        self.jesus_button.configure(state='disabled' if blocked else 'normal')
        self.supportive_button.configure(state='disabled' if blocked else 'normal')
        self.send_button.configure(state='disabled' if blocked else 'normal')
        self.input.configure(state='disabled' if blocked else 'normal')
        if self.runner.busy:
            state = 'Generating' if self.runner.ready else 'Loading local 8-bit model'
        else:
            state = 'Ready' if self.runner.ready else 'Model unloaded'
        if self.controller.pending is not None:
            state += ' · awaiting your approval'
        if self.jesus_mode.get():
            state += ' · Jesus mode (AI simulation)'
        self.status.set(state + ' · offline')

    def send(self, event=None):
        if self.runner.busy or self.controller.pending is not None:
            return 'break'
        message = self.input.get('1.0', 'end').strip()
        try:
            self.controller.supportive = self.supportive.get()
            self.controller.jesus_mode = self.jesus_mode.get()
            messages = self.controller.begin(message)
            self.runner.touch()
            self.runner.submit(messages)
        except ValueError as error:
            self.write('System', str(error), 'system')
            return 'break'
        self.input.delete('1.0', 'end')
        self.write('You', message, 'user')
        self.refresh_controls()
        return 'break'

    def poll(self):
        if self.closed:
            return
        result = self.runner.poll()
        if result:
            kind, content = result
            if kind == 'answer':
                try:
                    action, value = self.controller.receive(content)
                    if action == 'read':
                        self.runner.submit(self.controller.messages())
                    elif action == 'reply':
                        speaker = 'Jesus mode (AI simulation)' if self.controller.jesus_mode else 'Assistant'
                        self.write(speaker, value)
                    else:
                        self.show_proposal(value)
                except ValueError as error:
                    self.controller.pending = None
                    self.remove_approval()
                    self.write('WARNING', f'This response cannot be approved.\n{error}\n'
                               'Nothing was saved. This action has been rejected; ask for a corrected action.', 'warning')
                except (TypeError, sqlite3.Error):
                    self.controller.pending = None
                    self.remove_approval()
                    self.write('WARNING', 'The response could not be safely processed. This action has been rejected '
                               'and cannot be approved. Nothing was saved. Try a simpler request or clear chat.', 'warning')
            elif kind == 'error':
                self.write('System', content, 'system')
            elif kind == 'unloaded':
                self.write('System', 'Idle timeout reached. Model unloaded.', 'system')
        self.refresh_controls()
        self.root.after(100, self.poll)

    def show_proposal(self, preview):
        self.write('Assistant', preview)
        self.chat.configure(state='normal')
        self.approval_frame = ttk.Frame(self.chat, padding=4)
        ttk.Button(self.approval_frame, text='I approve', command=self.approve).pack(side='left', padx=4)
        ttk.Button(self.approval_frame, text='Cancel', command=self.cancel).pack(side='left', padx=4)
        self.chat.window_create('end', window=self.approval_frame)
        self.chat.insert('end', '\n\n')
        self.chat.configure(state='disabled')
        self.chat.see('end')

    def remove_approval(self):
        if self.approval_frame is not None:
            self.approval_frame.destroy()
            self.approval_frame = None

    def approve(self):
        # This callback is the sole application call site for Controller.approve().
        self.remove_approval()
        try:
            result = self.controller.approve()
        except (ValueError, sqlite3.Error):
            result = 'The change failed or the record changed. Nothing saved; request a fresh preview.'
        self.write('System', result, 'system')
        self.refresh_controls()

    def cancel(self):
        self.controller.cancel()
        self.remove_approval()
        self.write('System', 'Change cancelled. Nothing saved.', 'system')
        self.refresh_controls()

    def unload(self):
        was_busy = self.runner.busy
        self.runner.stop()
        if was_busy:
            self.controller.history.append({'role': 'user', 'content': 'APPLICATION RESULT: Generation cancelled.'})
        self.write('System', 'Model unloaded. GPU worker stopped.', 'system')
        self.refresh_controls()

    def clear(self):
        self.runner.stop()
        self.controller.clear()
        self.remove_approval()
        self.chat.configure(state='normal')
        self.chat.delete('1.0', 'end')
        self.chat.configure(state='disabled')
        self.input.configure(state='normal')
        self.input.delete('1.0', 'end')
        self.write('System', 'Chat cleared and model unloaded. Saved records remain.', 'system')
        if self.jesus_mode.get():
            self.write('System', SIMULATION_NOTICE, 'system')
        self.refresh_controls()

    def change_jesus_mode(self):
        self.controller.jesus_mode = self.jesus_mode.get()
        notice = SIMULATION_NOTICE if self.jesus_mode.get() else 'Jesus mode turned off.'
        self.write('System', notice, 'system')
        self.refresh_controls()

    def show_compendium(self):
        """Browse the built-in quotations without loading the model or saving data."""
        window = tk.Toplevel(self.root)
        window.configure(background=COLORS['background'])
        window.title('Christ quotations — offline compendium')
        window.geometry('850x650')
        display = dark_scrolled_text(window, wrap='word', font=TEXT_FONT)
        display.pack(fill='both', expand=True, padx=8, pady=8)
        display.insert('end', compendium_text())
        display.configure(state='disabled')

    def browse(self):
        window = tk.Toplevel(self.root)
        window.configure(background=COLORS['background'])
        window.title('Saved records — read only')
        window.geometry('750x560')
        bar = ttk.Frame(window, padding=8)
        bar.pack(fill='x')
        query = ttk.Entry(bar)
        query.pack(side='left', fill='x', expand=True)
        display = dark_scrolled_text(window, wrap='word', state='disabled', font=TEXT_FONT)
        display.pack(fill='both', expand=True, padx=8, pady=8)
        pagination = {'offset': 0, 'next': None}
        detail_bar = ttk.Frame(window, padding=8)
        detail_bar.pack(fill='x')
        ttk.Label(detail_bar, text='Record ID:').pack(side='left')
        ident = ttk.Entry(detail_bar, width=10)
        ident.pack(side='left', padx=8)

        def show(value):
            display.configure(state='normal')
            display.delete('1.0', 'end')
            display.insert('end', json.dumps(value, indent=2, ensure_ascii=False))
            display.configure(state='disabled')

        def search(reset=True):
            if reset:
                pagination['offset'] = 0
            try:
                result = self.store.search(query.get(), pagination['offset'])
                pagination['next'] = result['next_offset']
                show(result)
            except ValueError as error:
                show(str(error))

        def next_page():
            if pagination['next'] is not None:
                pagination['offset'] = pagination['next']
                search(False)

        def details():
            try:
                show(self.store.get(int(ident.get())))
            except ValueError:
                show('Enter an existing numeric record ID.')

        ttk.Button(bar, text='Search / first page', command=search).pack(side='left', padx=6)
        ttk.Button(bar, text='Next', command=next_page).pack(side='left')
        ttk.Button(detail_bar, text='Show record and notes', command=details).pack(side='left')
        search()

    def callback_error(self, exception_type, exception, traceback):
        # Never use Tkinter's default traceback printer for potentially private data.
        self.runner.stop()
        self.controller.pending = None
        self.remove_approval()
        self.write('System', 'An internal error interrupted this operation. Model unloaded. '
                   'Inspect saved records before retrying.', 'system')
        self.refresh_controls()

    def close(self):
        self.closed = True
        self.runner.stop()
        self.controller.clear()
        self.store.close()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path(__file__).resolve().parent / 'models' / 'qwen3-4b-int8')
    parser.add_argument('--data-dir', type=Path, default=default_data_dir())
    parser.add_argument('--idle-seconds', type=int, default=120)
    args = parser.parse_args()
    if not 10 <= args.idle_seconds <= 3600:
        parser.error('--idle-seconds must be between 10 and 3600')
    root = tk.Tk()
    try:
        Application(root, args.model.resolve(), args.data_dir.resolve(), args.idle_seconds)
    except Exception:
        messagebox.showerror('Cannot start', 'Cannot open the application or its local database. Check folder permissions.')
        root.destroy()
        return
    root.mainloop()


if __name__ == '__main__':
    mp.freeze_support()
    main()
