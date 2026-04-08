"""Patch requests to work with Sparkle/mihomo fake-ip DNS environment.

Problem: Sparkle uses fake-ip DNS (198.18.0.0/15), which breaks Python's
urllib3 proxy CONNECT tunneling. curl works because it delegates DNS to
the proxy, but urllib3 resolves DNS locally first.

Solution: Use httpx with proxy, which handles this correctly. Retry on failure.
"""

import time
import requests
import requests.models

_patched = False


def apply():
    global _patched
    if _patched:
        return

    try:
        import httpx
    except ImportError:
        return

    proxies = requests.utils.getproxies()
    proxy_url = proxies.get("https") or proxies.get("http")
    if not proxy_url:
        return

    _orig_request = requests.Session.request

    def _make_client():
        return httpx.Client(
            proxy=proxy_url,
            timeout=httpx.Timeout(30, connect=15),
            follow_redirects=True,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=5),
        )

    _clients = {"c": _make_client()}

    def _patched_request(self, method, url, **kwargs):
        max_retries = 3
        last_exc = None
        for attempt in range(max_retries):
            try:
                client = _clients["c"]
                headers = {}
                if self.headers:
                    headers.update(dict(self.headers))
                if kwargs.get("headers"):
                    headers.update(dict(kwargs["headers"]))

                r = client.request(
                    method, url,
                    headers=headers,
                    params=kwargs.get("params"),
                    content=kwargs.get("data"),
                    json=kwargs.get("json"),
                )
                resp = requests.models.Response()
                resp.status_code = r.status_code
                resp.headers = dict(r.headers)
                resp._content = r.content
                resp.url = str(r.url)
                resp.encoding = r.encoding
                resp.request = requests.models.PreparedRequest()
                resp.request.url = url
                resp.request.method = method
                return resp
            except Exception as e:
                last_exc = e
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))
                    try:
                        _clients["c"].close()
                    except Exception:
                        pass
                    _clients["c"] = _make_client()

        # All retries failed, try original as last resort
        try:
            return _orig_request(self, method, url, **kwargs)
        except Exception:
            raise last_exc

    requests.Session.request = _patched_request
    _patched = True
