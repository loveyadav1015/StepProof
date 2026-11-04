#!/usr/bin/env python3
"""Local development HTTP server for docs/site with CORS headers and dual port support."""

from __future__ import annotations

import argparse
import http.server
import os
import sys
from pathlib import Path


class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler that adds CORS headers to all responses."""

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With, Content-Type, Accept, Range")
        self.send_header("Access-Control-Expose-Headers", "Content-Length, Content-Range")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()


def serve(port: int = 8000, bind: str = "0.0.0.0", directory: Path | str = "docs/site") -> None:
    target_dir = Path(directory).resolve()
    if not target_dir.exists():
        print(f"Directory not found: {target_dir}. Running scripts/build_site.py first...")
        import subprocess
        subprocess.run([sys.executable, "scripts/build_site.py"], check=True)

    # Change cwd to directory so SimpleHTTPRequestHandler serves it cleanly
    os.chdir(target_dir)

    # Try binding to port; if busy, try port + 1
    max_retries = 5
    current_port = port
    server = None

    for i in range(max_retries):
        try:
            server = http.server.ThreadingHTTPServer((bind, current_port), CORSRequestHandler)
            break
        except OSError as e:
            if e.errno in (98, 48):  # Address already in use
                print(f"⚠️  Port {current_port} is in use, trying port {current_port + 1}...")
                current_port += 1
            else:
                raise

    if server is None:
        print(f"❌ Could not bind to port in range {port} - {port + max_retries - 1}")
        sys.exit(1)

    print("=" * 60)
    print("STEPPROOF STATIC SITE SERVER")
    print("=" * 60)
    print(f"Directory:    {target_dir}")
    print("CORS Headers: Access-Control-Allow-Origin: *")
    print(f"URL:          http://localhost:{current_port}")
    if bind == "0.0.0.0":
        print(f"Network URL:  http://127.0.0.1:{current_port}")
    print("=" * 60)
    print("Press Ctrl+C to stop the server.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping site server.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve docs/site with CORS headers.")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--bind", "-b", type=str, default="0.0.0.0", help="Address to bind to (default: 0.0.0.0)")
    parser.add_argument("--dir", "-d", type=str, default="docs/site", help="Directory to serve (default: docs/site)")

    args = parser.parse_args()
    serve(port=args.port, bind=args.bind, directory=args.dir)


if __name__ == "__main__":
    main()
