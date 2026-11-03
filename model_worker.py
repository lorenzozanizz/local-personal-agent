"""Model lifecycle. This child process is the only process that imports CUDA."""
import multiprocessing as mp
from contextlib import redirect_stderr, redirect_stdout
import os
from pathlib import Path
import socket
import time


def disable_network():
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1',
                      HF_HUB_DISABLE_PROGRESS_BARS='1', TQDM_DISABLE='1',
                      TOKENIZERS_PARALLELISM='false')

    def blocked(*args, **kwargs):
        raise RuntimeError('Network access is disabled in the model worker.')

    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.socket.sendto = blocked
    socket.create_connection = blocked
    socket.getaddrinfo = blocked


def generate(model, tokenizer, messages):
    import torch
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                          tokenize=True, return_dict=True,
                                          return_tensors='pt').to('cuda:0')
    if inputs['input_ids'].shape[1] > 7168:
        raise ValueError('context')
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=768, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id,
                                 use_cache=True)
    new_tokens = outputs[0, inputs['input_ids'].shape[1]:]
    # A truncated action is never safe to interpret as a complete proposal.
    if len(new_tokens) >= 768 and new_tokens[-1].item() != tokenizer.eos_token_id:
        raise ValueError('length')
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def worker(connection, model_path):
    disable_network()
    # Replace Python streams rather than their file descriptors. On Windows,
    # dup2 can leave a console TextIOWrapper attached to an incompatible handle.
    # Keep the sink open for the entire lifetime of the redirected streams.
    with open(os.devnull, 'w', encoding='utf-8') as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            run_worker(connection, model_path)


def run_worker(connection, model_path):
    """Run with working non-console streams and progress bars disabled."""
    stage = 'checking the model folder'
    initialized = False
    try:
        model_path = str(Path(model_path).expanduser().resolve())
        if not (Path(model_path) / 'config.json').is_file():
            raise FileNotFoundError('No config.json in the selected model folder.')
        stage = 'importing the inference libraries'
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.utils import logging
        logging.set_verbosity_error()
        logging.disable_progress_bar()
        if not torch.cuda.is_available():
            connection.send(('error', 'CUDA is unavailable. Check your NVIDIA driver and PyTorch installation.'))
            return
        if torch.cuda.get_device_capability(0) < (7, 5):
            connection.send(('error', 'LLM.int8 requires an NVIDIA Turing GPU or newer.'))
            return
        stage = 'loading the tokenizer'
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True,
                                                  trust_remote_code=False)
        stage = 'loading the 8-bit checkpoint'
        model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, device_map={'': 0}, torch_dtype=torch.float16)
        if not getattr(model, 'is_loaded_in_8bit', False):
            connection.send(('error', 'This model is not 8-bit. Run prepare_model.py first.'))
            return
        stage = 'initializing inference'
        model.eval()
        initialized = True
        connection.send(('ready', ''))
        while True:
            messages = connection.recv()
            if messages is None:
                return
            try:
                answer = generate(model, tokenizer, messages)
                connection.send(('answer', answer))
                del messages, answer
            except ValueError:
                connection.send(('error', 'The context or reply is too long. Clear chat or request a smaller result.'))
            except Exception:
                connection.send(('error', 'Generation failed. The model will unload; check available VRAM and package compatibility.'))
                return
    except (EOFError, BrokenPipeError):
        pass
    except Exception as error:
        try:
            # Before initialization, no chat or record content has been received.
            # Show setup exceptions in the UI only; never save a diagnostic log.
            if not initialized:
                detail = (f'Model setup failed while {stage}.\n'
                          f'Model folder: {model_path}\n'
                          f'{type(error).__name__}: {str(error)[:2000]}')
            else:
                detail = 'The model worker failed. Model unloaded; no diagnostic log was saved.'
            connection.send(('error', detail))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        connection.close()


class ModelRunner:
    """Nonblocking interface, polled only by Tk's main thread."""
    def __init__(self, model_path, idle_seconds=120):
        self.model_path = str(model_path)
        self.idle_seconds = idle_seconds
        self.process = None
        self.connection = None
        self.ready = False
        self.busy = False
        self.queued = None
        self.last_activity = time.monotonic()
        self.started = 0.0

    def touch(self):
        self.last_activity = time.monotonic()

    def submit(self, messages):
        if self.busy:
            raise ValueError('A response is already in progress.')
        self.busy = True
        if self.process is None:
            context = mp.get_context('spawn')
            self.connection, child = context.Pipe()
            self.process = context.Process(target=worker, args=(child, self.model_path), daemon=True)
            try:
                self.process.start()
                child.close()
            except Exception:
                child.close()
                self.stop()
                raise ValueError('Cannot start the model process.') from None
            self.started = time.monotonic()
            self.queued = messages
        else:
            self.connection.send(messages)

    def poll(self):
        if self.process is None:
            return None
        try:
            if self.connection.poll():
                kind, value = self.connection.recv()
                if kind == 'ready':
                    self.ready = True
                    self.connection.send(self.queued)
                    self.queued = None
                else:
                    self.busy = False
                if kind == 'error':
                    self.stop()
                return kind, value
        except (EOFError, OSError):
            self.stop()
            return 'error', 'The model process stopped. Check the local model and GPU memory.'
        if not self.process.is_alive():
            self.stop()
            return 'error', 'The model process exited unexpectedly.'
        # Bound a stuck dependency load without blocking the interface forever.
        if not self.ready and time.monotonic() - self.started > 300:
            self.stop()
            return 'error', 'Model loading timed out after five minutes.'
        if not self.busy and time.monotonic() - self.last_activity >= self.idle_seconds:
            self.stop()
            return 'unloaded', ''
        return None

    def stop(self):
        if self.process is not None:
            if self.process.pid is not None:
                self.process.terminate()
                self.process.join(timeout=0.5)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout=0.5)
            self.process.close()
        if self.connection is not None:
            self.connection.close()
        self.process = self.connection = self.queued = None
        self.ready = self.busy = False