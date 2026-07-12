from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from src.churn_model import (
    DEFAULT_MODEL_PATH,
    InputValidationError,
    ModelArtifactNotFoundError,
    predict_payload,
)


PROJECT_ROOT = Path(__file__).resolve().parent
WEB_ROOT = PROJECT_ROOT / "web"
STATIC_ROOT = WEB_ROOT / "static"
RESULTS_ROOT = PROJECT_ROOT / "results"


class BonusAppHandler(BaseHTTPRequestHandler):
    server_version = "BonusChurnHTTP/1.0"

    def do_GET(self) -> None:
        path = urlparse(self.path).path

        if path in {"/", "/index.html"}:
            self._serve_file(WEB_ROOT / "index.html")
            return

        if path == "/api/health":
            self._write_json(
                {
                    "status": "ok",
                    "model_available": DEFAULT_MODEL_PATH.exists(),
                    "model_path": str(DEFAULT_MODEL_PATH.relative_to(PROJECT_ROOT)),
                }
            )
            return

        if path.startswith("/static/"):
            self._serve_safe_file(STATIC_ROOT, path.removeprefix("/static/"))
            return

        if path.startswith("/results/"):
            self._serve_safe_file(RESULTS_ROOT, path.removeprefix("/results/"))
            return

        self._write_json({"error": "Rota nao encontrada."}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/predict":
            self._write_json({"error": "Rota nao encontrada."}, HTTPStatus.NOT_FOUND)
            return

        try:
            payload = self._read_json()
            prediction = predict_payload(payload)
        except ValueError as exc:
            if isinstance(exc, InputValidationError):
                self._write_json(
                    {"error": "Entrada invalida.", "fields": exc.errors},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            self._write_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        except ModelArtifactNotFoundError as exc:
            self._write_json(
                {
                    "error": str(exc),
                    "setup": "Execute `python scripts/train_bonus_model.py` depois de gerar data/cleaned_dataset.csv.",
                },
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return
        except Exception as exc:  # noqa: BLE001 - surface a local dev error as JSON.
            self._write_json({"error": f"Falha ao prever: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._write_json(prediction)

    def _read_json(self) -> dict[str, object]:
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length > 64_000:
            raise ValueError("Payload muito grande.")

        raw_body = self.rfile.read(content_length).decode("utf-8")
        if not raw_body:
            return {}
        parsed = json.loads(raw_body)
        if not isinstance(parsed, dict):
            raise ValueError("Envie um objeto JSON.")
        return parsed

    def _serve_safe_file(self, root: Path, relative_path: str) -> None:
        requested = (root / unquote(relative_path)).resolve()
        root = root.resolve()
        if root not in requested.parents and requested != root:
            self._write_json({"error": "Caminho invalido."}, HTTPStatus.BAD_REQUEST)
            return
        self._serve_file(requested)

    def _serve_file(self, file_path: Path) -> None:
        if not file_path.exists() or not file_path.is_file():
            self._write_json({"error": "Arquivo nao encontrado."}, HTTPStatus.NOT_FOUND)
            return

        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        content = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _write_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        content = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interface web local para a fase bonus.")
    parser.add_argument("--host", default="127.0.0.1", help="Host do servidor local.")
    parser.add_argument("--port", default=8000, type=int, help="Porta do servidor local.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    server = ThreadingHTTPServer((args.host, args.port), BonusAppHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Interface web disponivel em {url}")
    print(f"Modelo: {'ok' if DEFAULT_MODEL_PATH.exists() else 'pendente'}")
    print("Use Ctrl+C para encerrar.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor encerrado.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
