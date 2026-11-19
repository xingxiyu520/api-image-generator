#!/usr/bin/env python3
"""Generate and edit images through a configured API."""

from __future__ import annotations

import argparse
import base64
import ipaddress
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


OPENAI_PROVIDER = "openai-compatible"
REPLICATE_PROVIDER = "replicate"
USER_AGENT = "api-image-generator/2.0"
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
IMAGE_EXTENSIONS = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}


class ImageGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Job:
    prompt: str
    input_images: tuple[Path, ...] = ()
    count: int = 1
    model: str | None = None
    size: str | None = None
    negative_prompt: str | None = None
    seed: int | None = None


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    key: str = ""
    base_url: str = ""
    edit_url: str = ""
    model: str = ""
    replicate_image_field: str = "image"


def positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, choices=[OPENAI_PROVIDER, REPLICATE_PROVIDER])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--batch-file", type=Path, help="UTF-8 JSON array of image jobs")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--input-image", action="append", default=[], help="Reference image path; repeat for multiple references")
    inputs.add_argument("--input-dir", type=Path, help="Edit each PNG, JPEG or WebP in this directory individually")
    parser.add_argument("--count", type=positive_int, default=1, help="Number of separate requests per job")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--size")
    parser.add_argument("--negative-prompt")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timeout", type=positive_int, default=120)
    parser.add_argument("--env-file", type=Path, help="Load provider configuration from this dotenv file")
    parser.add_argument("--mock-response", type=Path, help="Decode a local JSON response without calling the provider")
    args = parser.parse_args()
    if args.batch_file and (args.input_image or args.input_dir):
        parser.error("with --batch-file, set input_images separately in each job")
    return args


def read_env_file(path: Path) -> dict[str, str]:
    values = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.removeprefix("export ").partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not key:
            raise ImageGenerationError(f"Invalid .env line {line_number}: expected KEY=VALUE")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values.setdefault(key, value)
    return values


def settings(env_file: Path | None) -> dict[str, str]:
    if env_file is not None:
        return {**read_env_file(env_file.expanduser()), **os.environ}
    candidates = (Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env")
    dotenv = next((path for path in candidates if path.is_file()), None)
    return {**(read_env_file(dotenv) if dotenv else {}), **os.environ}


def require_setting(values: dict[str, str], name: str) -> str:
    value = values.get(name)
    if not value:
        raise ImageGenerationError(f"Missing required environment variable: {name}")
    return value


def checked_url(url: str, name: str) -> urllib.parse.SplitResult:
    try:
        parsed = urllib.parse.urlsplit(url)
        if (parsed.scheme in {"http", "https"} and parsed.hostname
                and not (parsed.username or parsed.password or parsed.fragment)):
            return parsed
    except ValueError:
        pass
    raise ImageGenerationError(f"{name} must be an HTTP(S) URL without embedded credentials or a fragment")


def is_loopback(hostname: str) -> bool:
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def load_config(args: argparse.Namespace) -> ProviderConfig:
    values = settings(args.env_file)
    if args.provider == REPLICATE_PROVIDER:
        return ProviderConfig(
            args.provider, key=require_setting(values, "REPLICATE_API_TOKEN"),
            model=values.get("REPLICATE_MODEL", ""),
            replicate_image_field=values.get("REPLICATE_IMAGE_FIELD", "image"),
        )
    base_url = require_setting(values, "IMAGE_API_BASE_URL")
    edit_url = values.get("IMAGE_API_EDIT_URL", "")
    base = checked_url(base_url, "IMAGE_API_BASE_URL")
    edit = checked_url(edit_url, "IMAGE_API_EDIT_URL") if edit_url else base
    key = values.get("IMAGE_API_KEY", "")
    if not key and not all(is_loopback(endpoint.hostname or "") for endpoint in (base, edit)):
        key = require_setting(values, "IMAGE_API_KEY")
    return ProviderConfig(args.provider, key, base_url, edit_url, values.get("IMAGE_API_MODEL", ""))


def reference_image(raw_path: str | Path, directory: Path) -> Path:
    path = (directory / Path(raw_path).expanduser()).resolve()
    if path.suffix.lower() not in IMAGE_TYPES or not path.is_file():
        raise ImageGenerationError(f"Reference must be an existing PNG, JPEG or WebP file: {path}")
    return path


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.expanduser().read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ImageGenerationError(f"Invalid UTF-8 JSON in {path}") from exc


def load_jobs(args: argparse.Namespace) -> list[Job]:
    default = Job(args.prompt or "", count=args.count, model=args.model, size=args.size,
                  negative_prompt=args.negative_prompt, seed=args.seed)
    if args.batch_file:
        manifest = args.batch_file.expanduser().resolve()
        items = read_json(manifest)
        if not isinstance(items, list) or not items:
            raise ImageGenerationError("Batch file must contain a nonempty JSON array")
        return [manifest_job(item, default, manifest.parent, index) for index, item in enumerate(items, 1)]
    if not default.prompt.strip():
        raise ImageGenerationError("Prompt cannot be empty")
    if args.input_dir:
        directory = args.input_dir.expanduser().resolve()
        paths = sorted(
            (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_TYPES),
            key=lambda path: (path.name.casefold(), path.name),
        )
        if not paths:
            raise ImageGenerationError(f"No PNG, JPEG or WebP files found in {directory}")
        return [replace(default, input_images=(reference_image(path, directory),)) for path in paths]
    return [replace(default, input_images=tuple(reference_image(path, Path.cwd()) for path in args.input_image))]


def manifest_job(item: Any, default: Job, directory: Path, index: int) -> Job:
    fields = {"prompt", "input_images", "count", "model", "size", "negative_prompt", "seed"}
    if not isinstance(item, dict) or item.keys() - fields:
        raise ImageGenerationError(f"Job {index} must be an object using only {', '.join(sorted(fields))}")
    prompt = item.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ImageGenerationError(f"Job {index} needs a nonempty prompt")
    count = item.get("count", default.count)
    if type(count) is not int or count < 1:
        raise ImageGenerationError(f"Job {index} count must be a positive integer")
    seed = item.get("seed", default.seed)
    if seed is not None and type(seed) is not int:
        raise ImageGenerationError(f"Job {index} seed must be an integer")
    images = item.get("input_images", [])
    if not isinstance(images, list) or not all(isinstance(path, str) and path for path in images):
        raise ImageGenerationError(f"Job {index} input_images must be an array of file paths")
    options = {field: item.get(field, getattr(default, field)) for field in ("model", "size", "negative_prompt")}
    if any(value is not None and not isinstance(value, str) for value in options.values()):
        raise ImageGenerationError(f"Job {index} model, size and negative_prompt must be strings")
    return Job(prompt, tuple(reference_image(path, directory) for path in images), count, seed=seed, **options)


class NoProviderRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


PROVIDER_HTTP = urllib.request.build_opener(NoProviderRedirect())


def http_json(method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float) -> dict[str, Any]:
    headers = {"User-Agent": USER_AGENT, **headers}
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with PROVIDER_HTTP.open(request, timeout=timeout) as response:
            parsed = json.load(response)
    except urllib.error.HTTPError as exc:
        exc.close()
        hints = {
            400: " Check the model and request parameters.",
            401: " Check the API credential.",
            403: " Check the credential permissions and model access.",
            404: " Check the configured endpoint and model.",
            405: " Check whether the endpoint supports this operation.",
            413: " The request is too large.",
            422: " Check the input fields and image format.",
            429: " Check the provider rate limit or quota.",
        }
        hint = " Configure the final endpoint URL." if 300 <= exc.code < 400 else hints.get(exc.code, "")
        raise ImageGenerationError(f"Provider returned HTTP {exc.code}.{hint}") from exc
    except TimeoutError as exc:
        raise ImageGenerationError("Timed out calling the image API") from exc
    except urllib.error.URLError as exc:
        raise ImageGenerationError(f"Could not connect to the image API ({type(exc.reason).__name__})") from exc
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ImageGenerationError("Image API returned a non-JSON response") from exc
    if not isinstance(parsed, dict):
        raise ImageGenerationError("Image API response must be a JSON object")
    return parsed


def json_bytes(data: dict[str, Any]) -> bytes:
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


def openai_endpoint(config: ProviderConfig, edit: bool) -> str:
    if edit and config.edit_url:
        return config.edit_url
    parsed = urllib.parse.urlsplit(config.base_url)
    path = parsed.path.rstrip("/")
    for suffix in ("/images/generations", "/images/edits"):
        if path.endswith(suffix):
            path = path.removesuffix(suffix)
            break
    operation = "edits" if edit else "generations"
    return parsed._replace(path=f"{path}/images/{operation}").geturl()


def encode_multipart(fields: dict[str, Any], images: tuple[Path, ...]) -> tuple[str, bytes]:
    boundary = f"imagegen-{uuid4().hex}"
    chunks = []
    for key, value in fields.items():
        chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode("utf-8"))
    image_field = "image" if len(images) == 1 else "image[]"
    for index, path in enumerate(images, 1):
        suffix = path.suffix.lower()
        header = (f'--{boundary}\r\nContent-Disposition: form-data; name="{image_field}"; '
                  f'filename="reference-{index}{suffix}"\r\nContent-Type: {IMAGE_TYPES[suffix]}\r\n\r\n')
        chunks.extend((header.encode("ascii"), path.read_bytes(), b"\r\n"))
    chunks.append(f"--{boundary}--\r\n".encode("ascii"))
    return f"multipart/form-data; boundary={boundary}", b"".join(chunks)


def optional_fields(job: Job) -> dict[str, Any]:
    fields = {}
    if job.negative_prompt:
        fields["negative_prompt"] = job.negative_prompt
    if job.seed is not None:
        fields["seed"] = job.seed
    return fields


def call_openai(config: ProviderConfig, job: Job, model: str, timeout: int) -> dict[str, Any]:
    payload = {"model": model, "prompt": job.prompt, "n": 1, **optional_fields(job)}
    if job.size:
        payload["size"] = job.size
    headers = {"Accept": "application/json"}
    if config.key:
        headers["Authorization"] = f"Bearer {config.key}"
    if job.input_images:
        headers["Content-Type"], body = encode_multipart(payload, job.input_images)
    else:
        headers["Content-Type"], body = "application/json", json_bytes(payload)
    return http_json("POST", openai_endpoint(config, bool(job.input_images)), headers, body, timeout)


def replicate_prediction_url(model: str) -> tuple[str, dict[str, str]]:
    if "/" in model:
        encoded = "/".join(urllib.parse.quote(part, safe="") for part in model.split("/", 1))
        return f"https://api.replicate.com/v1/models/{encoded}/predictions", {}
    return "https://api.replicate.com/v1/predictions", {"version": model}


def parse_size(size: str | None) -> dict[str, int]:
    if not size or "x" not in size.lower():
        return {}
    width, height = size.lower().split("x", 1)
    try:
        return {"width": int(width), "height": int(height)}
    except ValueError:
        return {}


def call_replicate(config: ProviderConfig, job: Job, model: str, timeout: int) -> dict[str, Any]:
    url, root_fields = replicate_prediction_url(model)
    model_input = {"prompt": job.prompt, **optional_fields(job), **parse_size(job.size)}
    if job.input_images:
        image = job.input_images[0]
        data = base64.b64encode(image.read_bytes()).decode("ascii")
        model_input[config.replicate_image_field] = f"data:{IMAGE_TYPES[image.suffix.lower()]};base64,{data}"
    headers = {
        "Authorization": f"Token {config.key}", "Content-Type": "application/json",
        "Accept": "application/json", "Prefer": "wait=10",
    }
    deadline = time.monotonic() + timeout
    prediction = http_json("POST", url, headers, json_bytes({**root_fields, "input": model_input}), timeout)
    while prediction.get("status") in {"starting", "processing"}:
        urls = prediction.get("urls")
        poll_url = urls.get("get") if isinstance(urls, dict) else None
        if not isinstance(poll_url, str) or not poll_url:
            raise ImageGenerationError("Replicate response did not include a prediction status URL")
        parsed = checked_url(poll_url, "Prediction status URL")
        if parsed.scheme != "https" or parsed.netloc != "api.replicate.com":
            raise ImageGenerationError("Unexpected Replicate prediction status host")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ImageGenerationError("Timed out waiting for Replicate prediction")
        time.sleep(min(2, remaining))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ImageGenerationError("Timed out waiting for Replicate prediction")
        prediction = http_json("GET", poll_url, headers, None, remaining)
    if prediction.get("error") or prediction.get("status") in {"failed", "canceled"}:
        raise ImageGenerationError("Replicate prediction failed. Check the model status in Replicate.")
    return prediction


def looks_like_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def find_image_reference(value: Any) -> tuple[str, str] | None:
    if isinstance(value, dict):
        for key in ("b64_json", "image", "base64", "url"):
            item = value.get(key)
            if isinstance(item, str) and item:
                if looks_like_url(item):
                    return "url", item
                if key != "url" or item.startswith("data:image/"):
                    return "base64", item
        for key in ("data", "output", "images", "result", "image"):
            if found := find_image_reference(value.get(key)):
                return found
    elif isinstance(value, list):
        for item in value:
            if found := find_image_reference(item):
                return found
    elif isinstance(value, str):
        if looks_like_url(value):
            return "url", value
        if value.startswith("data:image/"):
            return "base64", value
    return None


def image_extension(data: bytes, content_type: str = "") -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    extension = IMAGE_EXTENSIONS.get(content_type.split(";", 1)[0].strip().lower())
    if extension and data:
        return extension
    raise ImageGenerationError("Provider response did not contain a supported image")


def save_result(response: Any, directory: Path, timeout: int) -> tuple[Path, str | None]:
    reference = find_image_reference(response)
    if not reference:
        raise ImageGenerationError("No image URL or base64 image found in the provider response")
    kind, value = reference
    remote_url = None
    content_type = ""
    if kind == "url":
        parsed = checked_url(value, "Image download URL")
        request = urllib.request.Request(value, headers={"Accept": "image/*", "User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as download:
                data, content_type = download.read(), download.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            exc.close()
            raise ImageGenerationError(f"Image download returned HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ImageGenerationError("Could not download the generated image") from exc
        remote_url = parsed._replace(query="").geturl()
    else:
        if value.startswith("data:image/"):
            header, _, value = value.partition(",")
            content_type = header.removeprefix("data:").split(";", 1)[0]
        try:
            data = base64.b64decode(value, validate=True)
        except ValueError as exc:
            raise ImageGenerationError("Provider response contained invalid base64 image data") from exc
    extension = image_extension(data, content_type)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = directory / f"generated-{timestamp}-{uuid4().hex[:12]}{extension}"
    with path.open("xb") as output:
        try:
            output.write(data)
        except OSError:
            output.close()
            path.unlink(missing_ok=True)
            raise
    return path, remote_url


def execute(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    jobs = load_jobs(args)
    mock_response = read_json(args.mock_response) if args.mock_response else None
    config = ProviderConfig(args.provider, model="mock-model") if args.mock_response else load_config(args)
    models = [job.model or config.model for job in jobs]
    if not all(models):
        raise ImageGenerationError("Missing model. Pass --model or set IMAGE_API_MODEL or REPLICATE_MODEL")
    if config.provider == REPLICATE_PROVIDER and any(len(job.input_images) > 1 for job in jobs):
        raise ImageGenerationError("Replicate edits support one reference per job in this skill")
    directory = args.output_dir.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    is_batch = bool(args.batch_file or args.input_dir or args.count > 1)
    results = []
    failures = 0
    for job_number, (job, model) in enumerate(zip(jobs, models), 1):
        for variation in range(1, job.count + 1):
            metadata = {"job": job_number, "variation": variation, "mode": "edit" if job.input_images else "generate"}
            try:
                if args.mock_response:
                    response = mock_response
                    reference = find_image_reference(response)
                    if reference and reference[0] == "url":
                        raise ImageGenerationError("Mock responses must use base64 to keep tests offline")
                elif config.provider == REPLICATE_PROVIDER:
                    response = call_replicate(config, job, model, args.timeout)
                else:
                    response = call_openai(config, job, model, args.timeout)
                if config.provider == REPLICATE_PROVIDER and isinstance(response, dict) and "output" in response:
                    response = response["output"]
                path, remote_url = save_result(response, directory, args.timeout)
                result = {"saved_path": str(path), "provider": config.provider, "model": model, "remote_url": remote_url}
                results.append({**metadata, **result} if is_batch else result)
            except (ImageGenerationError, OSError) as exc:
                if not is_batch:
                    raise
                failures += 1
                results.append({**metadata, "error": str(exc)})
    if not is_batch:
        return results[0], 0
    return {"succeeded": len(results) - failures, "failed": failures, "results": results}, int(failures > 0)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    try:
        result, status = execute(parse_args())
    except (ImageGenerationError, OSError, UnicodeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
