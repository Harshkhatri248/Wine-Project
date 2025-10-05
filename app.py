from flask import Flask, render_template, Response
import concurrent.futures
import threading
from typing import Optional
import logging
import os
import json


def create_app(config: Optional[dict] = None):
    # Global error handler to log and display exceptions
    @app.errorhandler(Exception)
    def handle_exception(e):
        import traceback
        tb = traceback.format_exc()
        app.logger.error(f"Unhandled Exception: {e}\n{tb}")
        # Show the traceback in the browser for debugging (remove in production)
        return f"<h2>Internal Server Error</h2><pre>{tb}</pre>", 500
    """Application factory for WSGI servers (e.g. Waitress).

    The factory attaches a single-thread ThreadPoolExecutor and a future to the
    returned app so background model runs are managed on the app object.
    """
    app = Flask(__name__, template_folder='templates')

    if config:
        app.config.update(config)

    # Import model inside factory so failures are attached to the app and not at import time
    try:
        import model
    except Exception:
        model = None

    app.model = model
    app.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    app.future_lock = threading.Lock()
    app.future = None

    # Ensure logs directory exists
    logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, 'model_runs.log')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)

    def _submit_model():
        with app.future_lock:
            if app.future is None or app.future.done():
                if app.model and hasattr(app.model, 'run_model'):
                    app.logger.info('Submitting background model run')
                    def _wrap():
                        try:
                            result = app.model.run_model()
                            app.logger.info('Model run finished: %s', str(result))
                            return result
                        except Exception as e:
                            app.logger.exception('Model run failed')
                            raise

                    app.future = app.executor.submit(_wrap)
                else:
                    f = concurrent.futures.Future()
                    f.set_result('No model available')
                    app.future = f
        return app.future

    @app.route('/')
    def home():
        fut = _submit_model()
        if fut.running():
            output = 'Model is running (background). Refresh to see result.'
        elif fut.done():
            try:
                output = fut.result()
            except Exception as e:
                output = f'Model error: {e}'
            # clear finished future so the user can re-run
            with app.future_lock:
                app.future = None
        else:
            output = 'Model started (background). Refresh to see result.'
        return render_template('index.html', output=output)

    @app.route('/ping')
    def ping():
        return Response('ok', mimetype='text/plain')

    @app.route('/status')
    def status():
        with app.future_lock:
            fut = app.future
            if fut is None:
                state = 'idle'
                result = None
            elif fut.running():
                state = 'running'
                result = None
            elif fut.done():
                state = 'done'
                try:
                    result = fut.result()
                except Exception as e:
                    result = f'error: {e}'
            else:
                state = 'unknown'
                result = None
        payload = {'state': state, 'result': result}
        return Response(json.dumps(payload), mimetype='application/json')

    return app


# Create a module-level app for local runs (python app.py)
app = create_app()


if __name__ == '__main__':
    # Bind explicitly and disable the debug reloader when starting as a background process on Windows
    app.run(host='127.0.0.1', port=5000, debug=False)