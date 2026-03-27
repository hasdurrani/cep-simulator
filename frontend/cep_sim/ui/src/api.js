const BASE = "/api";

async function req(method, path, body) {
  const res = await fetch(BASE + path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

export const api = {
  configs:  ()      => req("GET",  "/configs"),
  setup:    (body)  => req("POST", "/setup",    body),
  simulate: (body)  => req("POST", "/simulate", body),
  baseline: (sessionId, brandId) => req("GET", `/baseline/${sessionId}${brandId ? `?brand_id=${brandId}` : ""}`),
  artifactUrl: (sessionId, filename) => `${BASE}/artifacts/${sessionId}/${filename}`,
};
